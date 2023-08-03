#!/bin/bash
usage() {
  printf "Entrypoint to push model to the HF hub. If the model was trained with adapters, will
  create a separate branch for the adapters and a main branch with the merged adapters.\n\n\
\tUsage: $0 -d RESULTS_DIR -r HF_REPO_NAME [-b BASE_MODEL]\n" 1>&2
  exit 1
}

while getopts "b:d:r:" opt; do
  case $opt in
    b)
      BASE_MODEL="$OPTARG"
      ;;
    d)
      RESULTS_DIR="$OPTARG"
      ;;
    r)
      HF_REPO_NAME="$OPTARG"
      ;;
    \?)
      echo "Unrecognized options. For usage, see below."
      usage
      ;;
  esac
done

if [ -z "${RESULTS_DIR}" ] || [ -z "${HF_REPO_NAME}" ]; then
    echo "The following options are required: -d RESULTS_DIR, -r HF_REPO_NAME!"
    usage
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"

source .venv/bin/activate
# Create dir where we store all hub models
mkdir -p hub_models
cd hub_models

# Git init
huggingface-cli repo create --type model -y "$HF_REPO_NAME"
git clone "https://huggingface.co/BramVanroy/$HF_REPO_NAME"
cd "$HF_REPO_NAME"

LOCAL_REPO_DIR="$PWD"

git lfs install
huggingface-cli lfs-enable-largefiles .

cp "$RESULTS_DIR/*" .

# if trained with LoRA/adapters:
if [ -f "adapter_model.bin" ]; then
    if [ -z "${BASE_MODEL}" ]; then
        echo "When your results contain adapters, you must specify the BASE_MODEL (-b)!"
        usage
    fi
    # Git adapters
    git checkout -b adapters

    git add .
    git commit -m "init adapters"
    git push --set-upstream origin adapters

    # Git main
    git checkout main
    git merge adapters
    ## Merge adapters
    cd "$SCRIPT_DIR"
    python merge_adapters.py --base_model_name "$BASE_MODEL" --adapter_model_name "$LOCAL_REPO_DIR" --output_name "$LOCAL_REPO_DIR"

    ## Push main
    cd "$LOCAL_REPO_DIR"
    git rm adapter_*
    git add .
    git commit -m "init main model"
else
    git add .
    git commit -m "init main model"
fi

git push
