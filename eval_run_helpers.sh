#!/usr/bin/env bash

# Shared model discovery and GPU reservation helpers for the evaluation scripts.

scan_model_folders() {
    local raw_folder folder model_dir found
    for raw_folder in "${folders[@]}"; do
        folder="${raw_folder%/}"
        [[ -n "$folder" ]] || folder=/
        [[ -d "$folder" ]] || { echo "Folder not found: $folder" >&2; return 2; }

        found=0
        for model_dir in "$folder"/*; do
            if [[ -d "$model_dir" && -f "$model_dir/config.json" ]]; then
                models+=("$model_dir")
                ((found += 1))
            fi
        done
        echo "Found $found model(s) in $folder"
    done
}

dedupe_models() {
    local model
    local -a unique=()
    local -A seen=()

    for model in "${models[@]}"; do
        [[ -n "$model" ]] || { echo "Model names must not be empty." >&2; return 2; }
        if [[ -z "${seen[$model]:-}" ]]; then
            unique+=("$model")
            seen[$model]=1
        fi
    done
    models=("${unique[@]}")
}

model_tag() {
    local name="${1%/}"
    name="${name##*/}"
    printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

setup_gpu_pool() {
    local driver_lib gpu_list gpu used program
    local -A seen=()

    # Temporary workaround for this machine's NVIDIA library mismatch.
    driver_lib=/home/zf28/align/.nvidia-595.84/compute-lib
    if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
        [[ -e "$driver_lib/libcuda.so.1" ]] || {
            echo "Missing NVIDIA compatibility libraries: $driver_lib" >&2
            return 1
        }
        export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    for program in nvidia-smi flock; do
        command -v "$program" >/dev/null 2>&1 || {
            echo "$program is required to coordinate GPU availability." >&2
            return 1
        }
    done

    if (( ${#gpus[@]} == 0 )); then
        gpu_list="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
        if [[ -n "$gpu_list" ]]; then
            IFS=',' read -r -a gpus <<< "$gpu_list"
        else
            mapfile -t gpus < <(
                nvidia-smi --query-gpu=index --format=csv,noheader,nounits
            )
        fi
    fi
    (( ${#gpus[@]} > 0 )) || {
        echo "No GPUs found. Set GPUS or use --gpus 0,1." >&2
        return 1
    }

    max_used_mb="${MAX_USED_MB:-500}"
    poll_seconds="${GPU_POLL_SECONDS:-10}"
    [[ "$max_used_mb" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
        echo "MAX_USED_MB must be nonnegative and GPU_POLL_SECONDS must be positive." >&2
        return 2
    }

    for gpu in "${gpus[@]}"; do
        [[ -z "${seen[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; return 2; }
        seen[$gpu]=1
        used=$(gpu_memory_used "$gpu") || used=""
        [[ "$used" =~ ^[0-9]+$ ]] || {
            echo "Unable to query GPU $gpu with nvidia-smi." >&2
            return 1
        }
    done
}

gpu_memory_used() {
    nvidia-smi --id="$1" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

gpu_compute_pids() {
    nvidia-smi --id="$1" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

gpu_lock_file() {
    local name="${1//[^A-Za-z0-9._-]/_}"
    printf '/tmp/align_eval_gpu_%s.lock' "$name"
}

gpu_is_empty() {
    local used pids
    used=$(gpu_memory_used "$1") || return 1
    pids=$(gpu_compute_pids "$1") || return 1
    [[ "$used" =~ ^[0-9]+$ ]] && (( used < max_used_mb )) && [[ -z "$pids" ]]
}

gpu_is_ready() {
    flock -n "$(gpu_lock_file "$1")" true 2>/dev/null && gpu_is_empty "$1"
}

gpu_status() {
    local used pids
    used=$(gpu_memory_used "$1") || used="unknown"
    pids=$(gpu_compute_pids "$1") || pids="unknown"
    printf '%s MB used, PIDs: %s' "$used" "${pids:-none}"
}

wait_until_gpu_empty() {
    local gpu="$1" announced=0
    until gpu_is_empty "$gpu"; do
        if (( announced == 0 )); then
            echo "GPU $gpu is not empty ($(gpu_status "$gpu")); waiting."
            announced=1
        fi
        sleep "$poll_seconds"
    done
}

# Hold this GPU's shared lock for the lifetime of the command.
run_on_gpu() {
    local gpu="$1"
    shift
    (
        flock 9
        wait_until_gpu_empty "$gpu"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@"
    ) 9>"$(gpu_lock_file "$gpu")"
}

held_lock_fds=()

release_gpu_group() {
    local fd
    for fd in "${held_lock_fds[@]}"; do
        eval "exec ${fd}>&-"
    done
    held_lock_fds=()
}

try_reserve_gpu_group() {
    local gpu fd
    local -a selected=("$@") acquired=()

    for gpu in "${selected[@]}"; do
        exec {fd}>"$(gpu_lock_file "$gpu")"
        if flock -n "$fd"; then
            acquired+=("$fd")
        else
            eval "exec ${fd}>&-"
            held_lock_fds=("${acquired[@]}")
            release_gpu_group
            return 1
        fi
    done

    for gpu in "${selected[@]}"; do
        if ! gpu_is_empty "$gpu"; then
            held_lock_fds=("${acquired[@]}")
            release_gpu_group
            return 1
        fi
    done

    held_lock_fds=("${acquired[@]}")
    available_gpu_group=$(IFS=,; echo "${selected[*]}")
}

# Wait for and exclusively reserve COUNT empty GPUs.
reserve_gpu_group() {
    local count="$1" gpu announced=0
    local -a selected

    while true; do
        selected=()
        for gpu in "${gpus[@]}"; do
            if gpu_is_ready "$gpu"; then
                selected+=("$gpu")
                (( ${#selected[@]} == count )) && break
            fi
        done

        if (( ${#selected[@]} == count )) && try_reserve_gpu_group "${selected[@]}"; then
            return
        fi

        if (( announced == 0 )); then
            echo "Waiting for $count empty GPU(s)..."
            announced=1
        fi
        sleep "$poll_seconds"
    done
}
