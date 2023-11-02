import json
import argparse
import platform, multiprocessing
import os
import time

from nsql.nsql_exec import Executor, NeuralDB
from utils.normalizer import post_process_sql
from utils.utils import majority_vote
from utils.evaluator import Evaluator

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def worker_execute(pid, args, dataset, nsql_dict, keys):
    result_dict = dict()
    n_total_samples, n_correct_samples = 0, 0
    for eid, data_item in enumerate(dataset):
        eid = str(eid)
        if eid not in nsql_dict:
            continue
        print(f"Process#{pid}: eid {eid}, wtq-id {data_item['id']}")
        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text']
        n_total_samples += 1
        table = data_item['table']
        title = table['page_title']
        executor = Executor(args, keys)

        exec_answer_list = []
        nsql_exec_answer_dict = dict()
        for idx, (nsql, logprob) in enumerate(nsql_dict[eid]['nsqls']):
            print(f"Process#{pid}: eid {eid}, original_id {data_item['id']}, executing program#{idx}, logprob={logprob}")
            try:
                if nsql in nsql_exec_answer_dict:
                    exec_answer = nsql_exec_answer_dict[nsql]
                else:
                    db = NeuralDB(
                        tables=[{"title": title, "table": table}]
                    )
                    nsql = post_process_sql(
                        sql_str=nsql,
                        df=db.get_table_df(),
                        process_program_with_fuzzy_match_on_db=args.process_program_with_fuzzy_match_on_db,
                        table_title=title
                    )
                    exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)
                    if isinstance(exec_answer, str):
                        exec_answer = [exec_answer]
                    nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                print(f"Process#{pid}: Execution error {e}")
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)

            if nsql_dict[eid].get('exec_answers', None) is None:
                nsql_dict[eid]['exec_answers'] = []
            nsql_dict[eid]['exec_answers'].append(exec_answer)

        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=nsql_dict[eid]['nsqls'],
            pred_answer_list=exec_answer_list,
            allow_none_and_empty_answer=args.allow_none_and_empty_answer,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )
        
        result_dict[eid]['pred_answer'] = pred_answer
        result_dict[eid]['nsql'] = pred_answer_nsqls
        gold_answer = data_item['answer_text']
        score = Evaluator().evaluate(
            pred_answer,
            gold_answer,
            dataset=args.dataset,
            question=result_dict[eid]['question']
        )
        n_correct_samples += score
        print(f'Process#{pid}: pred answer: {pred_answer}')
        print(f'Process#{pid}: gold answer: {gold_answer}')
        if score == 1:
            print(f'Process#{pid}: Correct!')
        else:
            print(f'Process#{pid}: Wrong.')
        print(f'Process#{pid}: Accuracy: {n_correct_samples}/{n_total_samples}')

    with open(os.path.join(args.save_dir, f"{pid}.json"), 'w') as f:
        json.dump(nsql_dict, f, indent=4)

    return result_dict


def process_program_data(data):
    nsql_dict = {}
    for eid, data_dict in data.items():
        if data[eid]['generations']:
            nsqls = data[eid]['generations']
        else:
            nsqls = [['<dummy program>', 0.]]
        nsql_dict[eid] = {'nsqls': nsqls}
    return nsql_dict

def split_data_by_processes(nsql_dict, n_processes):
    nsql_dict_group = [dict() for _ in range(n_processes)]
    for idx, eid in enumerate(nsql_dict.keys()):
        nsql_dict_group[idx % n_processes][eid] = nsql_dict[eid]
    return nsql_dict_group

def execute_programs_multiprocessing(args, dataset, nsql_dict_group, keys):
    result_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        worker_results.append(pool.apply_async(worker_execute, args=(
            pid,
            args,
            dataset,
            nsql_dict_group[pid],
            keys
        ))

    for r in worker_results:
        worker_result_dict = r.get()
        result_dict.update(worker_result_dict)
    pool.close()
    pool.join()
    return result_dict

def evaluate_results(result_dict, args):
    n_correct_samples = 0
    for eid, item in result_dict.items():
        pred_answer, gold_answer = item['pred_answer'], item['gold_answer']
        n_correct_samples += Evaluator().evaluate(
            pred_answer,
            gold_answer,
            dataset=args.dataset,
            question=result_dict[eid]['question']
        )
    return n_correct_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'tab_fact'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/qa_retrieve_pool/qa_retrieve_pool.json')
    parser.add_argument('--input_program_file', type=str,
                        default='binder_program_wikitq_test_chatgpt.json')
    parser.add_argument('--output_program_execution_file', type=str,
                        default='binder_program_wikitq_test_chatgpt_exec.json')
    parser.add_argument('--n_processes', type=str, default=1)
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--use_majority_vote', action='store_false',
                        help='Whether use majority vote to determine the prediction answer.')
    parser.add_argument('--allow_none_and_empty_answer', action='store_true',
                        help='Whether regarding none and empty executions as a valid answer.')
    parser.add_argument('--allow_error_answer', action='store_true',
                        help='Whether regarding error execution as a valid answer.')
    parser.add_argument('--answer_placeholder', type=int, default=0,
                        help='Placeholder answer if execution error occurs.')
    parser.add_argument('--vote_method', type=str, default='simple',
                        choices=['simple', 'prob', 'answer_biased'])
    parser.add_argument('--answer_biased', type=int, default=None,
                        help='The answer to be biased w. answer_biased_weight in majority vote.')
    parser.add_argument('--answer_biased_weight', type=float, default=None,
                        help='The weight of the answer to be biased in majority vote.')
    parser.add_argument('--process_program_with_fuzzy_match_on_db', action='store_false',
                        help='Whether use fuzzy match with db and program to improve on program.')

    args = parser.parse_args()

    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = load_data_split(args.dataset, args.dataset_split)

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

    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    nsql_dict = process_program_data(data)

    nsql_dict_group = split_data_by_processes(nsql_dict, args.n_processes)s
    result_dict = execute_programs_multiprocessing(args, dataset, nsql_dict_group, keys)
    n_correct_samples = evaluate_results(result_dict, args)
    print(f'Overall Accuracy: {n_correct_samples}/{len(result_dict)}')

    with open(os.path.join(args.save_dir, args.output_program_execution_file), 'w') as f:
        json.dump(result_dict, f)