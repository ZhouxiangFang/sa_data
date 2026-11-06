#!/usr/bin/env bash
# Download models and the evaluation data into the Hugging Face cache (HF_HOME,
# /scratch/zf28/huggingface) before jobs start: vLLM's download lock is in the node's /tmp, so
# jobs on different nodes would otherwise download the same files into the shared cache together.
#   range/prefetch.sh [--models_only] [REPO_ID...]
# Default models: the git20k models, WildGuard and the embedders. Weights are fetched the way
# vLLM loads them: safetensors only when the repo has them, never Llama's original/ copy.
# Unless --models_only: every dataset the code loads, the test sets and IF benchmarks built
# through eval_safe/eval_if (so the datasets cache is built once), and IFBench's NLTK data.
# Works on the login node or as a CPU job:  GPUS=0 NAME=prefetch range/gpu.sh bash range/prefetch.sh
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

models_only=0
[[ ${1:-} == --models_only ]] && { models_only=1; shift; }
models=(FocusV857/llama3.1_git20k FocusV857/mistral_git20k FocusV857/olmo2_git20k
        FocusV857/qwen2.5_git20k allenai/wildguard BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B)
(( $# )) && models=("$@")
datasets=(allenai/wildguardmix allenai/wildjailbreak nvidia/Aegis-AI-Content-Safety-Dataset-2.0
          gretelai/gretel-safety-alignment-en-v1 Paul/XSTest bench-llm/or-bench google/IFEval)
(( models_only )) && datasets=()
echo "HF_HOME=$HF_HOME"

python - "${#models[@]}" "${models[@]}" "${datasets[@]}" <<'EOF'
import sys
from huggingface_hub import list_repo_files, snapshot_download

n = int(sys.argv[1])
repos = [(r, "model") for r in sys.argv[2:2 + n]] + [(r, "dataset") for r in sys.argv[2 + n:]]
failed = []
for repo, kind in repos:
    try:
        ignore = None
        if kind == "model":
            ignore = ["original/*", "*.pth", "*.pt", "*.gguf", "*.h5", "*.msgpack", "*.ot", "onnx/*"]
            if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
                ignore.append("*.bin")
        path = snapshot_download(repo, repo_type=kind, ignore_patterns=ignore)
        print(f"ok      {kind:7} {repo} -> {path}", flush=True)
    except Exception as error:  # e.g. a gated repo whose terms are not accepted
        print(f"FAILED  {kind:7} {repo}: {type(error).__name__}: {error}", flush=True)
        failed.append(repo)
sys.exit(1 if failed else 0)
EOF
status=$?
(( models_only )) && exit "$status"

python -c "import nltk; [nltk.download(p, quiet=True) for p in
           ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng')]" &&
    echo "ok      nltk punkt, punkt_tab, stopwords, averaged_perceptron_tagger_eng" || status=1
cd -- "$SA_DIR" || exit
python -c "
import eval_if, eval_safe
print('ok      IF benchmarks:', {k: len(v) for k, v in eval_if.load_benchmarks().items()})
sets = eval_safe.load_test_sets(eval_safe.TEST_SETS)
print('ok      test sets:', {k: len(v) for k, v in sets.items()})" 2>&1 | grep -v NVML || status=1
exit "$status"
