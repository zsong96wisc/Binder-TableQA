import os, pandas, json, pprint
from typing import Iterator, Optional

from dataset import download_unzip, get_pandas_table, get_sqlalchemy_table, na_values

# Download dataset
tabfact_hash = "948b5560e2f7f8c9139bd91c7f093346a2bb56a8"
tabfact_dir = f"{os.getcwd()}/Table-Fact-Checking-{tabfact_hash}"
tabfact_url = (
    f"https://github.com/wenhuchen/Table-Fact-Checking/archive/{tabfact_hash}.zip"
)
download_unzip("tabfact", tabfact_dir, tabfact_url)

data_dir = f"{tabfact_dir}/data/all_csv"
tabfact_json = f"{tabfact_dir}/tokenized_data/total_examples.json"


def get_pandas_schema(table_id: str, table_title: str) -> str:
    dfcsv = pandas.read_csv(
        f"{data_dir}/{table_id}",
        delimiter="#",
        header=0,
        na_values=na_values,
    )
    dfcsv.name = table_title
    return get_pandas_table(dfcsv)


def get_sqlalchemy_schema(table_id: str, table_title: str) -> str:
    dfcsv = pandas.read_csv(
        f"{data_dir}/{table_id}",
        delimiter="#",
        header=0,
        na_values=na_values,
    )
    dfcsv.name = table_title
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
    """Iterator for Dataset TabFact

    Args:
        table_list (list[str], optional): The list of requested tables, where empty list means all the tables in the dataset. Defaults to [].
        use_sql (bool, optional): Whether use sqlalchemy plan or pandas plan. Pandas plan will also returns a dataframe. Defaults to False.

    Yields:
        Iterator[tuple[Optional[pandas.DataFrame], str, list[str], list[str]]]: Return an iterator of [(data frame, optional), table_prompt, a list of questions, a list of answers]
    """
    with open(tabfact_json, "r") as jsonfile:
        json_data = json.load(jsonfile)
    if len(table_list) == 0:
        for table_id, values in json_data.items():
            yield *get_table(use_sql)(table_id, values[-1]), values[0], values[1]
    else:
        for table_id in table_list:
            values = json_data[table_id]
            yield *get_table(use_sql)(table_id, values[-1]), values[0], values[1]
