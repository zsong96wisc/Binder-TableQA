from dataset import Dataset

PANDAS_FUNC_TEMPLATE = "def check_statement(df):"
WIKITQ_FUNC_TEMPLATE = "def answer_question(df):"

a = """\
"""


def parse_pandas(gen: str, dataset: Dataset) -> str:
    gen_list = gen.strip().split("\n")
    program = []
    for g in gen_list:
        if len(program) > 0:
            if g[:4] == " " * 4 or g[:1] == "\t":
                program.append(g)
                continue
            else:
                return "\n".join(program)
        if g.strip() == (
            PANDAS_FUNC_TEMPLATE if dataset == Dataset.TABFACT else WIKITQ_FUNC_TEMPLATE
        ):
            program.append(g)
        else:
            continue
    if len(program) > 0:
        return "\n".join(program)
    raise Exception("Can't Execute")


# print(parse_pandas(a, Dataset.WIKITQ))
