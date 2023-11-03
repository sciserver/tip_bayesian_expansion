import json
import logging
import os

import pandas
import sqlalchemy as sqla


class MSSQLDatabase:
    # wraps a Microsoft SQL Server database
    #
    # TBD do we want to support CasJobs iso/as well as sqlalchemy?
    #

    @classmethod
    def from_url(cls, url):
        db_info = sqla.engine.url.make_url(url)
        return cls(
            {
                "host": db_info.host,
                "database": db_info.database,
                "user": db_info.username,
                "pwd": db_info.password,
            }
        )

    @classmethod
    def from_file(cls, path: str) -> "MSSQLDatabase":
        with open(path, "r") as f:
            config = json.load(f)

        return cls(
            user=config["user"],
            pwd=config["pwd"],
            host=config["host"],
            database=config["database"],
        )

    def __init__(self, user: str, pwd: str, host: str, database: str):
        self.url = sqla.URL.create(
            "mssql+pymssql",
            username=user,
            password=pwd,
            host=host,
            database=database,
            port=1433,
        )

        self.ENGINE = sqla.create_engine(self.url)

    def execute_query(self, sql):
        with self.ENGINE.connect() as conn:
            return pandas.read_sql(sqla.text(sql), conn)

    def execute_update(self, statement):
        with self.ENGINE.connect() as connection:
            with connection.begin():
                connection.execute(sqla.text(statement))

    def __create_engine(self):
        return sqla.create_engine(
            (
                f"mssql+pymssql://{self.AUTH['user']}:{self.AUTH['pwd']}@{self.SERVER}:1433/{self.DATABASE}"
                "?charset=utf8"
            )
        )
