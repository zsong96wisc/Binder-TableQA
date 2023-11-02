import os
import sys


ROOT_DIR = os.path.join(os.path.dirname(__file__))
if ROOT_DIR in sys.path:
    sys.path.insert(0, ROOT_DIR)

command = (
    f"export PYTHONPATH={ROOT_DIR}:$PYTHONPATH && "
    f"export TOKENIZERS_PARALLELISM=false && "
    f"python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq "
    f"--dataset_split test "
    f"--prompt_file templates/prompts/wikitq_binder.txt "
    f"--n_parallel_prompts 1 "
    f"--max_generation_tokens 512 "
    f"--temperature 0.4 "
    f"--sampling_n 20 "
    f"-v"
)

os.system(command)


