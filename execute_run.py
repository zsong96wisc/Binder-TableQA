import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

# wikitq nsql execution command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset wikitq \
 --dataset_split test \
 --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
 --input_program_file binder_program_wikitq_test.json \
 --output_program_execution_file binder_program_wikitq_test_exec.json \
 --vote_method simple
 """)

# tab_fact nsql execution command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset tab_fact \
 --dataset_split test \
 --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
 --input_program_file binder_program_tab_fact_test.json \
 --output_program_execution_file binder_program_tab_fact_test_exec.json \
 --allow_none_and_empty_answer \
 --vote_method answer_biased \
 --answer_biased 1 \
 --answer_biased_weight 3 \
 """)
