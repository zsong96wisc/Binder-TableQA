from urllib.request import urlretrieve
import zipfile, os, csv, re, pandas, json, time, pprint
from typing import Iterator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Download dataset
tabfact_hash = "948b5560e2f7f8c9139bd91c7f093346a2bb56a8"
tabfact_dir = f"{os.getcwd()}/Table-Fact-Checking-{tabfact_hash}"
tabfact_url = (
    f"https://github.com/wenhuchen/Table-Fact-Checking/archive/{tabfact_hash}.zip"
)
if not os.path.exists(tabfact_dir):
    print("Downloading TabFact")
    urlretrieve(tabfact_url, "tabfact.zip")
    with zipfile.ZipFile("tabfact.zip", "r") as zip_file:
        zip_file.extractall()


def sample_data(col: pandas.Series, sep: str = "|"):
    sample_len = min(10, len(col))
    samples = list(col.sample(n=sample_len))
    str_samples = [str(s) for s in samples]
    return "|".join(str_samples)


data_dir = f"{tabfact_dir}/data/all_csv"
tabfact_json = f"{tabfact_dir}/tokenized_data/total_examples.json"


pandas_template = """>>> df.name = NAME
>>> df.dtype
DTYPE
"""


def get_pandas_table(table_id: str, table_title: str) -> str:
    dfcsv = pandas.read_csv(
        f"{data_dir}/{table_id}",
        delimiter="#",
        header=0,
        na_values=["n / a", "none", "-", "xx", "unknown"],
    )
    dtypes = dfcsv.dtypes
    print_dtypes = str(dtypes).split("\n")
    str_dtype_list = [
        f"{content}\t# e.g.,\t{sample_data(dfcsv[name])}"
        for name, content in zip(dtypes.keys(), print_dtypes[:-1])
    ]
    str_dtype_list.append(print_dtypes[-1])
    str_dtype = "\n".join(str_dtype_list)
    table_schema = pandas_template.replace("NAME", table_title)
    table_schema = table_schema.replace("DTYPE", str_dtype)
    return table_schema


# Create in-memory database
engine = create_engine("sqlite://", echo=True)


# Base class
class Base(DeclarativeBase):
    pass


sqlalchemy_template = """class Table(Base):
    __tablename__ = "NAME"
    index: Mapped[int] = mapped_column(primary_key=True)
    COLS
"""
col_template = "NAME: Mapped[TYPE] = mapped_column('ACTUAL') # e.g., EXPL"


def get_sqlalchemy_table(table_id: str, table_title: str) -> str:
    table_schema = sqlalchemy_template.replace("NAME", table_title)
    dfcsv = pandas.read_csv(
        f"{data_dir}/{table_id}",
        delimiter="#",
        header=0,
        na_values=["n / a", "none", "-", "xx", "unknown"],
    )
    dtypes = dfcsv.dtypes
    cols = []
    for i, (name, dtype) in enumerate(dtypes.items()):
        col = col_template.replace("NAME", f"col{i}")
        if dtype == "int64":
            col = col.replace("TYPE", "int")
        elif dtype == "float64":
            col = col.replace("TYPE", "float")
        else:
            col = col.replace("TYPE", "str")
        col = col.replace("ACTUAL", name)
        col = col.replace("EXPL", sample_data(dfcsv[name]))
        cols.append(col)
    table_schema = table_schema.replace("COLS", "\n    ".join(cols))
    exec(table_schema)
    Base.metadata.create_all(engine)
    dfcsv.to_sql(table_title, engine, if_exists="append")
    return table_schema


def get_table(use_sql: bool = False):
    if use_sql:
        return get_sqlalchemy_table
    else:
        return get_pandas_table


def dataset_iter(
    table_list: list[str] = [],
    *,
    use_sql: bool = False,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Iterator for Dataset TabFact

    Args:
        table_list (list[str], optional): The list of requested tables, where empty list means all the tables in the dataset. Defaults to [].
        use_sql (bool, optional): Whether use sqlalchemy plan or pandas plan. Defaults to False.

    Yields:
        Iterator[tuple[str, list[str], list[str]]]: Return an iterator of [table_prompt, a list of questions, a list of answers]
    """
    with open(tabfact_json, "r") as jsonfile:
        json_data = json.load(jsonfile)
    if len(table_list) == 0:
        for table_id, values in json_data.items():
            yield get_table(use_sql)(table_id, values[-1]), values[0], values[1]
    else:
        for table_id in table_list:
            values = json_data[table_id]
            yield get_table(use_sql)(table_id, values[-1]), values[0], values[1]
