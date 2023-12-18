from wikitq import dataset_iter
from prompt import gen_prompt_pandas, Dataset
from llm import call_llm
from parse import parse_pandas
import pandas as pd
import time


def answer_question(df: pd.DataFrame):
    raise Exception(df)


iter = 0
right = 0
start_time = time.time()
for df, table, stmt_list, ans_list in dataset_iter():
    for stmt, target in zip(stmt_list, ans_list):
        prompt = gen_prompt_pandas(table, stmt, Dataset.WIKITQ)
        iter += 1
        trying = 0
        while True:
            raw_gen = call_llm(prompt)
            trying += 1
            if trying > 3:
                end_time = time.time()
                print(
                    f"Time: {(end_time - start_time)/iter} Accuracy: {right}/{iter}={right/iter}"
                )
                break
            try:
                program = parse_pandas(raw_gen, Dataset.WIKITQ)
                exec(program)
                ans = answer_question(df)
                if str(ans) == target:
                    right += 1
                end_time = time.time()
                print(
                    f"Time: {(end_time - start_time)/iter} Accuracy: {right}/{iter}={right/iter}"
                )
                break
            except:
                continue
