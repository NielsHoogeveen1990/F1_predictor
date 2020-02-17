from formula1.utils import log_step_spark
import pyspark
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType


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

    races = races.withColumnRenamed("name", "GP")

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
              'positionText',
              'position',
              'number',
              'nationality',
              'round',
              'status',
              'forename',
              'surname',
              'laps',
              'milliseconds',
              'fastestLap',
              'rank',
              'fastestLapTime',
              'fastestLapSpeed',
              'constructorId',
              'name',
              'date'
              ]

    df = df.drop(*subset)
    return df


@log_step_spark
def sort_races(df):
    return df.orderBy(df.year.asc(), df.raceId.asc())


@log_step_spark
def average_finishing(df):
    window = Window.partitionBy('driverId').orderBy('year','raceId').rangeBetween(Window.unboundedPreceding, 0)

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
def sort_races1(df):
    return df.orderBy(df.year.asc(), df.raceId.asc())


@log_step_spark
def get_wins_per_circuit(df):
    window = Window.partitionBy('driverId', 'circuitId').orderBy('year','raceId').rangeBetween(Window.unboundedPreceding, 0)

    return df.withColumn('total_wins_circuit_tilldate', sf.sum('win').over(window))


@log_step_spark
def get_wins_total(df):
    window = Window.partitionBy('driverId').orderBy('year','raceId').rangeBetween(Window.unboundedPreceding, 0)

    return df.withColumn('total_wins_tilldate', sf.sum('win').over(window))


@log_step_spark
def lead_get_wins_per_circuit(df):
    window1 = Window.partitionBy('driverId', 'circuitId').orderBy('year', 'raceId')
    window2 = Window.partitionBy('driverId').orderBy('year', 'raceId')

    return (df.withColumn('total_wins_circuit_tilldate', sf.lag('total_wins_circuit_tilldate').over(window1))
            .withColumn('total_wins_tilldate', sf.lag('total_wins_tilldate').over(window2))
            )


@log_step_spark
def mean_last_5races(df):
    window = Window.partitionBy('driverId').orderBy('year').rangeBetween(-5, Window.currentRow)

    return df.withColumn('mean_last5_races', sf.avg('positionOrder').over(window))


@log_step_spark
def last_race_at_circuit(df):
    window = Window.partitionBy('driverId', 'circuitId').orderBy('year', 'raceId')

    return df.withColumn('last_race_at_circuit', sf.lag('positionOrder').over(window))


@log_step_spark
def result_previous_race(df):
    window = Window.partitionBy('driverId').orderBy('year', 'raceId')

    return df.withColumn('result_previous_race', sf.lag('positionOrder').over(window))




# @pandas_udf("statusId int,"
#             "raceId int,"
#             "driverId int,"
#             "grid int,"
#             "positionOrder int,"
#             "points double,"
#             "constructorRef string,"
#             "driverRef string,"
#             "year int,"
#             "circuitId int,"
#             "mean_position_overall double,"
#             "mean_position_percircuit double,"
#             "win int,"
#             "win_per_circuit int",
#             PandasUDFType.GROUPED_MAP)
# def get_wins_per_circuit(pdf):
#     # pdf is a pandas.DataFrame
#     win = pdf.win
#     return pdf.assign(win_per_circuit= win.shift(1).cumsum())


# @log_step_spark
# def get_wins_per_circuit_pipe(df):
#
#     return df.groupby('driverId', 'circuitId').apply(get_wins_per_circuit)


# vragen: hoe kun je expanding window gebruiken? rangebetween?
# hoe kun je na withColumn casten naar andere dtype?
# UDF: grouped map gebruiken voor dingen als mean_last_5races?
