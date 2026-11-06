#!/usr/bin/env bash
# Download the models and datasets of a run into the Hugging Face cache (~/.cache/huggingface,
# or HF_HOME) from the login node, before submitting: many array tasks starting at once
# would otherwise download the same files into the shared cache together.
#   range/prefetch.sh [REPO_ID...]   (default: the git20k models, WildGuard, the embedders
#                                     and every dataset the code loads)
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

models=(FocusV857/llama3.1_git20k FocusV857/mistral_git20k FocusV857/olmo2_git20k
        FocusV857/qwen2.5_git20k allenai/wildguard BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B)
datasets=(allenai/wildguardmix allenai/wildjailbreak nvidia/Aegis-AI-Content-Safety-Dataset-2.0
          gretelai/gretel-safety-alignment-en-v1 Paul/XSTest bench-llm/or-bench google/IFEval)
(( $# )) && { models=("$@"); datasets=(); }

python - "${#models[@]}" "${models[@]}" "${datasets[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download

n = int(sys.argv[1])
repos = [(r, "model") for r in sys.argv[2:2 + n]] + [(r, "dataset") for r in sys.argv[2 + n:]]
failed = []
for repo, kind in repos:
    try:
        path = snapshot_download(repo, repo_type=kind)
        print(f"ok      {kind:7} {repo} -> {path}", flush=True)
    except Exception as error:  # e.g. a gated repo whose terms are not accepted
        print(f"FAILED  {kind:7} {repo}: {type(error).__name__}: {error}", flush=True)
        failed.append(repo)
sys.exit(1 if failed else 0)
EOF
status=$?
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]" &&
    echo "ok      nltk punkt, punkt_tab"
exit "$status"
