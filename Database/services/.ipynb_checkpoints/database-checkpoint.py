from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert

import pandas as pd


class Database(object):
    con = None

    @staticmethod
    def initialize():  # Initializes Database (Mongodb must be already running on system)
        Database.engine = create_engine("sqlite:///financial_db.sqlite", echo=True)
        Database.engine.connect()
        Database.Base = declarative_base()
        Database.Session = sessionmaker(bind=Database.engine)

    @staticmethod
    def create_table(table):
        """
        Function creates a table in the database from the table object
        :param table: a Table object that inherits from Database.Base (instance of declarative base)
        :return: None
        """
        Database.Base.metadata.create_all(Database.engine, tables=[table.__table__])

    @staticmethod
    def copy_table(table):
        """
        Function creates a copy of the input table
        :param table_name: a string indicating which table to copy
        :return: None
        """

        metadata = MetaData(bind=Database.engine)
        if Database.engine.has_table(table.__tablename__ + "_prev"):
            # the table already exists, do something
            Database.engine.execute("DELETE FROM " + table.__tablename__ + "_prev")
        else:
            # Replace 'original_table' with the name of the table you want to duplicate
            original_table = Table(table.__tablename__, metadata, autoload=True)

            new_table = original_table.tometadata(metadata)
            new_table.name = table.__tablename__ + "_prev"  # set the name of the new table

            new_table.create()  # create the new table

        Database.engine.execute("INSERT INTO " + table.__tablename__ + "_prev SELECT * FROM " + table.__tablename__)

    @staticmethod
    def insert_from_df(table, df, indices=[0]):
        """
        Function inserts records from dataframe into the database table named table_name if it exists
        If there are duplicate primary keys (indexed in the first column), then the record is overwritten
        Use the backup table
        :param indices:
        :param table: table object
        :param df: dataframe
        :return: None
        """
        session = Database.Session()
        for index, row in df.iterrows():
            record = row.to_dict()
            insert_stmt = insert(table).values(record)
            do_update_stmt = insert_stmt.on_conflict_do_update(index_elements=[row.index[i] for i in indices],
                                                               set_=record)
            session.execute(
                do_update_stmt
            )
        session.commit()

    @staticmethod
    def insert_from_df_by_name(table_name, df, indices=[0]):
        """
        Function inserts records from dataframe into the database table named table_name if it exists
        If there are duplicate primary keys (indexed in the first column), then the record is overwritten
        Use the backup table
        :param indices:
        :param table_name: string
        :param df: dataframe
        :return: None
        """
        metadata = MetaData(bind=Database.engine)
        Database.insert_from_df(Table(table_name, metadata, autoload=True), df, indices)

    @staticmethod
    def pandas_insert(table, df, dtype=None):  # Inserts new record in db from a pandas dataframe
        df.to_sql(table, con=Database.engine, if_exists='append', dtype=dtype)

    @staticmethod
    def query(query):  # Returns all records from db.collection matching query
        with Database.engine.begin() as connection:
            return pd.read_sql(query, connection)  # query must be in JSON
        connection.close()

    @staticmethod
    def delete_records(table_name, where=""):  # Returns all records from db.collection matching query
        with Database.engine.begin() as connection:
            connection.execute("Delete from " + table_name + where)  # query must be in JSON

    @staticmethod
    def disconnect():
        Database.engine.dispose()
