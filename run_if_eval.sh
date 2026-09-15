#!/usr/bin/env bash
set -euo pipefail

# Temporary workaround: the running kernel still uses NVIDIA 595.84 while the
# system CUDA/NVML libraries have been upgraded to 595.91.07.
driver_lib=/tmp/nvidia-595.84-minimal

if grep -q '595\.84' /proc/driver/nvidia/version 2>/dev/null; then
    if [[ ! -e "$driver_lib/libcuda.so.1" ]]; then
        echo "Missing temporary NVIDIA 595.84 libraries: $driver_lib" >&2
        echo "Ask to recreate the workaround, or reboot to load driver 595.91.07." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$script_dir/if_eval.py" "$@"
