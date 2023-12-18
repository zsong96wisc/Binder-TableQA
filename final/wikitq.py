import os, pandas, json, pprint
from typing import Iterator, Optional

from dataset import download_unzip, get_pandas_table, get_sqlalchemy_table, na_values

# Download dataset
wikitq_hash = "7d455a5a707b96341ef72aff9428749d443d8aa9"
wikitq_dir = f"{os.getcwd()}/WikiTableQuestions-{wikitq_hash}"
wikitq_url = f"https://github.com/ppasupat/WikiTableQuestions/archive/{wikitq_hash}.zip"
download_unzip("wikitq", wikitq_dir, wikitq_url)


data_dir = f"{wikitq_dir}/docs/viewer/csv"
wikitq_json = f"{wikitq_dir}/docs/viewer/csv/tables.json"


def get_pandas_schema(table_id: str, table_title: str) -> str:
    dfcsv = pandas.read_html(
        f"{data_dir}/{table_id}",
        encoding="UTF-8",
        na_values=na_values,
    )[0]
    dfcsv.table_title_769 = table_title
    return get_pandas_table(dfcsv)


def get_sqlalchemy_schema(table_id: str, table_title: str) -> str:
    dfcsv = pandas.read_html(
        f"{data_dir}/{table_id}",
        encoding="UTF-8",
        na_values=na_values,
    )[0]
    dfcsv.table_title_769 = table_title
    return get_sqlalchemy_table(dfcsv)


def get_table(use_sql: bool = False):
    if use_sql:
        return get_sqlalchemy_schema
    else:
        return get_pandas_schema


def dataset_iter(
    table_list: list[str] = [],
    *,
    use_sql: bool = False,
) -> Iterator[tuple[Optional[pandas.DataFrame], str, list[str], list[str]]]:
    """Iterator for Dataset wikitq

    Args:
        table_list (list[str], optional): The list of requested tables, where empty list means all the tables in the dataset. Defaults to [].
        use_sql (bool, optional): Whether use sqlalchemy plan or pandas plan. Pandas plan will also returns a dataframe. Defaults to False.

    Yields:
        Iterator[tuple[Optional[pandas.DataFrame], str, list[str], list[str]]]: Return an iterator of [(data frame, optional), table prompt, a list of questions, a list of answers]
    """
    with open(wikitq_json, "r") as jsonfile:
        json_data = json.load(jsonfile)["tables"]
    if len(table_list) == 0:
        for first_index in json_data:
            for table_index in first_index[1]:
                table_id = f"{first_index[0]}-csv/{table_index}-clean.html"
                meta_qa = f"{first_index[0]}-csv/{table_index}-data.json"
                with open(f"{data_dir}/{meta_qa}", "r") as jsonfile:
                    json_qa = json.load(jsonfile)
                table_schema = get_table(use_sql)(
                    table_id, json_qa["metadata"]["title"]
                )
                questions = []
                answers = []
                for qas in json_qa["examples"]:
                    questions.append(qas["utterance"])
                    answers.append(qas["targetValue"])
                yield *table_schema, questions, answers
    else:
        for table_idx in table_list:
            first_index, table_index = tuple(table_idx.split("/"))
            table_id = f"{first_index}-csv/{table_index}-clean.html"
            meta_qa = f"{first_index}-csv/{table_index}-data.json"
            with open(f"{data_dir}/{meta_qa}", "r") as jsonfile:
                json_qa = json.load(jsonfile)
            table_schema = get_table(use_sql)(table_id, json_qa["metadata"]["title"])
            questions = []
            answers = []
            for qas in json_qa["examples"]:
                questions.append(qas["utterance"])
                answers.append(qas["targetValue"])
            yield *table_schema, questions, answers
