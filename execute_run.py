import os
import sys


ROOT_DIR = os.path.join(os.path.dirname(__file__))
if ROOT_DIR in sys.path:
    sys.path.insert(0, ROOT_DIR)

command = (
    f"export PYTHONPATH={ROOT_DIR}:$PYTHONPATH && "
    f"export TOKENIZERS_PARALLELISM=false && "
    f"python {ROOT_DIR}/scripts/execute_binder_program.py --dataset wikitq "
    f"--dataset_split test "
    f"--qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json "
    f"--input_program_file binder_program_wikitq_test_chatgpt.json "
    f"--output_program_execution_file binder_program_wikitq_test_exec.json "
    f"--vote_method simple "
)

os.system(command)

