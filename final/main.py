from tabfact import dataset_iter
from prompt import gen_prompt_pandas, Dataset
from llm import call_llm
from parse import parse_pandas
import pandas as pd


def check_statement(df: pd.DataFrame):
    raise Exception(df)


for df, table, stmt_list, ans_list in dataset_iter():
    i = 0
    for stmt, ans in zip(stmt_list, ans_list):
        prompt = gen_prompt_pandas(table, stmt, Dataset.TABFACT)
        while True:
            raw_gen = call_llm(prompt)
            program = parse_pandas(raw_gen)
            exec(program)
            try:
                print(check_statement(df), ans)
                break
            except:
                continue

        i += 1
        if i == 3:
            quit()
