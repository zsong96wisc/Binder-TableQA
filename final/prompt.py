from dataset import Dataset


def read_template(filename: str):
    with open(filename, "r") as fTemplate:
        prompt_template = fTemplate.read()
        return prompt_template


def gen_prompt_pandas(table: str, stmt_qs: str, dataset: Dataset) -> str:
    if dataset == Dataset.TABFACT:
        prompt = read_template("tabfact_pandas.txt")
        prompt = prompt.replace("[DATA_SCHEMA]", table)
        prompt = prompt.replace("[STATEMENT]", stmt_qs)
        return prompt
    elif dataset == Dataset.WIKITQ:
        prompt = read_template("wikitq_pandas.txt")
        prompt = prompt.replace("[DATA_SCHEMA]", table)
        prompt = prompt.replace("[QUESTION]", stmt_qs)
        return prompt
    else:
        raise NotImplementedError("Invalid Dataset")
