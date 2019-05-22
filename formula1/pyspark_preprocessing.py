from formula1.utils import log_step_spark
import pyspark
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql.window import Window


spark = (pyspark.sql.SparkSession.builder.getOrCreate())

sc = spark.sparkContext


@log_step_spark
def read_data():
    return spark.read.csv(
        "../data/results.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )


@log_step_spark
def join_constructors(df):
    constructors = spark.read.csv(
        "../data/constructors.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(constructors, on='constructorId', how='left')


@log_step_spark
def join_drivers(df):
    drivers = spark.read.csv(
        "../data/drivers.csv",
        encoding='ISO-8859-1',
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(drivers, on='driverId', how='left')


@log_step_spark
def join_races(df):
    races = spark.read.csv(
        "../data/races.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(races, on='raceId', how='left')


@log_step_spark
def join_status(df):
    status = spark.read.csv(
        "../data/status.csv",
        header=True,
        inferSchema=True,
        nanValue='NA'
    )

    return df.join(status, on='statusId', how='left')


@log_step_spark
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


@log_step_spark
def sort_races(df):
    return df.orderBy(df.year.asc(), df.raceId.asc())


@log_step_spark
def average_finishing(df):
    window = Window.partitionBy('driverId')

    return df.withColumn('mean_position_overall', sf.avg('positionOrder').over(window))


@log_step_spark
def average_finishing_percircuit(df):
    window = Window.partitionBy('driverId', 'circuitId')

    return df.withColumn('mean_position_percircuit', sf.avg('positionOrder').over(window))


@log_step_spark
def get_wins(df):
    return df.withColumn('win', sf.col('positionOrder') == 1)


@log_step_spark
def change_dtypes(df):
    return df.withColumn("win", sf.col("win").cast("integer"))


@log_step_spark
def get_wins_per_circuit(df):
    window = Window.partitionBy('driverId', 'circuitId').orderBy('raceId')

    return (df.withColumn('win_per_circuit'
            , sum(sf.col('win')))
            .over(window)
            )


# vragen: hoe kun je expanding window gebruiken? rangebetween?
# hoe kun je na withColumn casten naar andere dtype?
# UDF: grouped map gebruiken voor dingen als mean_last_5races?


