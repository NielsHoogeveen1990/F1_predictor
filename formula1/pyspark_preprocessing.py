from formula1.utils import log_step
import pyspark
import pandas as pd
import pyspark.sql.functions as sf

spark = (pyspark.sql.SparkSession.builder.getOrCreate())

sc = spark.sparkContext


def read_data():
    return spark.read.csv(
        "../data/results.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )


def join_constructors(df):
    constructors = spark.read.csv(
        "../data/constructors.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(constructors, on='constructorId', how='left')


def join_drivers(df):
    drivers = spark.read.csv(
        "../data/drivers.csv",
        encoding='ISO-8859-1',
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(drivers, on='driverId', how='left')

def join_races(df):
    races = spark.read.csv(
        "../data/races.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(races, on='raceId', how='left')


def join_status(df):
    status = spark.read.csv(
        "../data/status.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(status, on='statusId', how='left')


def drop_columns(df):
    subset = ['url',
              'resultId',
              'dob',
              '_c5',
              'code',
              'time',
              'positionText'
              ]

    df = df.drop(*subset)
    return df


def sort_races(df):
    return df.orderBy(df.year.asc(), df.raceId.asc())


# vragen: hoe kun je expanding window gebruiken?



