import sqlite3
from contextlib import closing
import os
import pandas as pd

from formula1.preprocessing import get_clean_app_df


# Create a database
def db_connection():
    sqlite_db = sqlite3.connect('db/f1.sqlite')
    return sqlite_db


def fill_db():
    df = get_clean_app_df()
    db_conn = db_connection()
    df.to_sql('f1_races', db_conn, if_exists='replace', index=False)


def run_query(db_conn, query, *args):
    with closing(db_conn.cursor()) as cursor:
        result = cursor.execute(query, args).fetchall()
        db_conn.commit() #commit actually pushes the changes to the database
    return result


def retrieve_race_data(driver1, driver2, circuit, year):
    db_conn = db_connection()
    result = run_query(db_conn,f'SELECT * FROM f1_races '
    f'WHERE (driverRef_left = "{driver1}" '
    f'AND driverRef_right= "{driver2}" '
    f'AND name_left= "{circuit}" '
    f'AND year_left= "{year}") ')

    print(result)


def retrieve_prediction_data(driver1, driver2, circuit, year):
    db_conn = db_connection()
    #result = run_query(db_conn,f'SELECT * FROM f1_races WHERE (constructorRef_left = "{constructor}" AND year_left= "{year}") ')

    return pd.read_sql(f'SELECT * FROM f1_races'
                       f' WHERE (driverRef_left = "{driver1}" '
                       f'AND driverRef_right= "{driver2}" '
                       f'AND name_left= "{circuit}" '
                       f'AND year_left= "{year}" ) ', db_conn)




