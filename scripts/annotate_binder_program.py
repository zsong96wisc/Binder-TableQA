import time
import json
import argparse
import copy
import os

from typing import List
from datasets import Dataset
from generation.generator import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB
from transformers import AutoTokenizer
import traceback


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'tab_fact'])
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo-instruct")
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--max_api_total_tokens', type=int, default=3800)
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_binder.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'create_table_select_3_full_table_w_all_passage_image',
                                 'create_table_select_3_full_table_w_gold_passage_image',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='nsql',
                        choices=['nsql', 'sql', 'answer', 'npython', 'python'])

    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    print("Parameters:")
    for arg in args.__dict__:
        print(arg + ": " + str(args.__dict__[arg]))
    st = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    pmt_parallel_n = args.n_parallel_prompts
    # we load a small subset of data(say 100) to test since it is expensive and time-consuming to use openai api
    subset_data = []
    count = 0
    for row in dataset:
        subset_data.append(dict(row))
        count += 1
        if count == 100:
            break

    transformed_data = {key: [] for key in subset_data[0].keys()}

    for row in subset_data:
        for key in row:
            transformed_data[key].append(row[key])
    dataset = Dataset.from_dict(transformed_data)
    print(f"dataset length is {len(dataset)} **********")
    gen = Generator(args, keys=keys)
    tkr = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    example_ids = list(range(len(dataset)))
    example_dict = dict()
    built_few_shot_prompts = []
    max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
    try:
        for example_id in example_ids:
            example_data_item = dataset[example_id]
            db = NeuralDB(
                tables=[{'title': example_data_item['table']['page_title'], 'table': example_data_item['table']}]
            )
            example_dict[example_id] = {
                'generations': [],
                'ori_data_item': copy.deepcopy(example_data_item)
            }

            example_data_item['table'] = db.get_table_df()
            example_data_item['title'] = db.get_table_title()
            few_shots_pmt = gen.build_few_shot_prompt_from_file(args.prompt_file, args.n_shots)
            gen_pmt = gen.build_generate_prompt(
                data_item=example_data_item,
                generate_type=(args.generate_type,)
            )
            prompt = few_shots_pmt + "\n\n" + gen_pmt

            while len(tkr.tokenize(prompt)) >= max_prompt_tokens: 
                args.n_shots -= 1
                few_shots_pmt = gen.build_few_shot_prompt_from_file(args.prompt_file,args.n_shots)
                prompt = few_shots_pmt + "\n\n" + gen_pmt

            print(f"Building prompt for example_id#{example_id}, original_id#{example_data_item['id']}")
            built_few_shot_prompts.append((example_id, prompt))
            if pmt_parallel_n > len(built_few_shot_prompts):
                continue
           
            print(f"Prompts ready with {len(built_few_shot_prompts)} parallels")
            response_dict = gen.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                example_dict[eid]['generations'] = g_pairs

            built_few_shot_prompts = []
    except Exception as e:
        traceback.print_exc()
        print(f"example_id#{example_id}, wtqid#{example_data_item['id']} generation error: {e}")

    if len(built_few_shot_prompts) > 0:
        response_dict = gen.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            example_dict[eid]['generations'] = g_pairs
    with open(os.path.join(args.save_dir, f'binder_program_{args.dataset}_{args.dataset_split}_chatgpt.json'), 'w') as f:
        json.dump(example_dict, f, indent=4)
