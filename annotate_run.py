import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

# wikitq nsql annotation command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq \
 --dataset_split test \
 --prompt_file templates/prompts/wikitq_binder.txt \
 --n_parallel_prompts 1 \
 --max_generation_tokens 512 \
 --temperature 0.4 \
 --sampling_n 20 \
 -v""")

# tab_fact nsql annotation command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset tab_fact \
 --dataset_split test \
 --prompt_file templates/prompts/tab_fact_binder.txt \
 --n_parallel_prompts 1 \
 --n_processes 1 \
 --n_shots 18 \
 --max_generation_tokens 256 \
 --temperature 0.6 \
 --sampling_n 50 \
 -v""")