PANDAS_FUNC_TEMPLATE = "def check_statement(df):"


def parse_pandas(gen: str) -> str:
    gen_list = gen.strip().split("\n")
    program = []
    for g in gen_list:
        if len(program) > 0:
            if g[:4] == " " * 4 or g[:1] == "\t":
                program.append(g)
                continue
            else:
                return "\n".join(program)
        if g.strip() == PANDAS_FUNC_TEMPLATE:
            program.append(g)
        else:
            continue
    raise Exception("Can't Execute")
