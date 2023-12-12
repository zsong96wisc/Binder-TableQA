from urllib.request import urlretrieve
import zipfile, os, pandas
from typing import Iterator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from enum import Enum


class Dataset(Enum):
    TABFACT = 1
    WIKITQ = 2


def download_unzip(name: str, ex_dir: str, url: str) -> None:
    if not os.path.exists(ex_dir):
        print(f"Downloading dataset {name}")
        urlretrieve(url, f"{name}.zip")
        with zipfile.ZipFile(f"{name}.zip", "r") as zip_file:
            zip_file.extractall()


def sample_data(col: pandas.Series, sep: str = "#"):
    sample_len = min(10, len(col))
    samples = list(col.sample(n=sample_len))
    str_samples = [str(s) for s in samples]
    return sep.join(str_samples)


na_values = ["n / a", "none", "-", "xx", "unknown", "—", "–", "--", "?"]

pandas_template = """>>> df.name = NAME
>>> df.dtype
DTYPE
"""


def get_pandas_table(dfcsv: pandas.DataFrame) -> str:
    table_title = dfcsv.name
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
    return dfcsv, table_schema


# Create in-memory database
engine = create_engine("sqlite://", echo=False)


# Base class
class Base(DeclarativeBase):
    pass


sqlalchemy_template = """class Table(Base):
    __tablename__ = "NAME"
    index: Mapped[int] = mapped_column(primary_key=True)
    COLS
"""
col_template = "NAME: Mapped[TYPE] = mapped_column('ACTUAL') # e.g., EXPL"


def get_sqlalchemy_table(dfcsv: pandas.DataFrame) -> str:
    table_title = dfcsv.name
    table_schema = sqlalchemy_template.replace("NAME", table_title)
    dtypes = dfcsv.dtypes
    cols = []
    isnull_col = dfcsv.isnull().any()
    for i, (name, dtype) in enumerate(dtypes.items()):
        col = col_template.replace("NAME", f"col{i}")
        if dtype == "int64":
            col = col.replace("TYPE", "Optional[int]" if isnull_col[name] else "int")
        elif dtype == "float64":
            col = col.replace(
                "TYPE", "Optional[float]" if isnull_col[name] else "float"
            )
        else:
            col = col.replace("TYPE", "Optional[str]" if isnull_col[name] else "str")
        col = col.replace("ACTUAL", name)
        col = col.replace("EXPL", sample_data(dfcsv[name]))
        cols.append(col)
    table_schema = table_schema.replace("COLS", "\n    ".join(cols))
    exec(table_schema)
    Base.metadata.create_all(engine)
    dfcsv.to_sql(table_title, engine, if_exists="append")
    return table_schema
