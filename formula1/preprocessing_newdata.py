import pandas as pd
import numpy as np
from formula1.utils import log_step

PATH = '../new_data'

@log_step
def read_data():
    names = ['resultId',
             'raceId',
             'driverId',
             'constructorId',
             'number',
             'grid',
             'position',
             'positionText',
             'positionOrder',
             'points',
             'laps',
             'time',
             'milliseconds',
             'fastestLap',
             'rank',
             'fastestLapTime',
             'fastestLapSpeed',
             'statusId']
    return pd.read_csv(f'{PATH}/results.csv', names = names)


@log_step
def merge_constructors(df):
    names = ['constructorId',
             'constructorRef',
             'name',
             'nationality',
             'url']
    constructors = pd.read_csv(f'{PATH}/constructors.csv', names = names)
    return df.merge(constructors[['constructorId','constructorRef']], left_on='constructorId', right_on='constructorId')


@log_step
def merge_drivers(df):
    names = ['driverId',
             'driverRef',
             'number',
             'code',
             'forename',
             'surname',
             'dob',
             'nationality',
             'url']

    drivers = pd.read_csv(f'{PATH}/driver.csv', encoding='ISO-8859-1', names = names)
    return df.merge(drivers, left_on='driverId', right_on='driverId')


@log_step
def merge_races(df):
    names = ['raceId',
             'year',
             'round',
             'circuitId',
             'name',
             'date',
             'time',
             'url']

    races = pd.read_csv(f'{PATH}/races.csv', names = names)
    return df.merge(races, left_on='raceId', right_on='raceId')


@log_step
def merge_status(df):
    names = ['statusId',
             'status']

    status = pd.read_csv(f'{PATH}/status.csv', names = names)
    return df.merge(status, left_on='statusId', right_on='statusId')


@log_step
def merge_driverstandings(df):
    names = ['driverStandingsId',
             'raceId',
             'driverId',
             'points',
             'position',
             'positionText',
             'wins']

    driverstandings = pd.read_csv(f'{PATH}/driver_standings.csv', names=names)
    return df.merge(driverstandings, how='left', left_on=['raceId', 'driverId'], right_on=['raceId', 'driverId'], suffixes=['_1', '_2'])


@log_step
def remove_columns(df):
    subset = ['resultId',
              'number_y',
              'code',
              'forename',
              'surname',
              'url_x',
              'time_y',
              'url_y',
              'driverStandingsId',
              'positionText_2',
            ]
    return df.drop(labels=subset, axis=1)


@log_step
def rename_columns(df):
    return df.rename(columns= {'position_2': 'championship_standing','position_1':'position', 'wins': 'wins_this_year','points_2':'WC_points_thisyear'})


@log_step
def sort_races(df):
    return df.sort_values(['year', 'raceId'], ascending=True)


@log_step
def average_finishing(df):
    return df.assign(
        mean_position_till_date = lambda df: df.groupby('driverId')['positionOrder']
        .transform(lambda df: df.shift(1).expanding().mean())
    )


@log_step
def average_finishing_percircuit(df):
    return df.assign(
        mean_position__percircuit_till_date = lambda df: df.groupby(['driverId','circuitId'])['positionOrder']
        .transform(lambda df: df.shift(1).expanding().mean())
    )


@log_step
def result_previous_race(df):
    return df.assign(
    result_previous_race = lambda df: df.groupby('driverId')['positionOrder'].shift(1)
    )


@log_step
def mean_last_5races(df):
    return df.assign(
        mean_last5_races = lambda df: df.groupby('driverId')['positionOrder']
        .transform(lambda df: df.shift(1).rolling(5).mean())
    )


@log_step
def last_race_at_circuit(df):
    return df.assign(
        last_race_at_circuit = lambda df: df.groupby(['driverId', 'circuitId'])['positionOrder']
        .transform(lambda df: df.shift(1))
    )


@log_step
def get_wins(df):
    return df.assign(
        win = lambda df: df['positionOrder'] == 1
    )


@log_step
def get_wins_per_circuit(df):
    return df.assign(
        win_per_circuit = lambda df: df.groupby(['driverId','circuitId'])['win']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def get_poles(df):
    return df.assign(
        pole = lambda df: df['grid'] == 1
    )


@log_step
def get_poles_per_circuit(df):
    return df.assign(
        poles_per_circuit = lambda df: df.groupby(['driverId','circuitId'])['pole']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def get_total_wins(df):
    return df.assign(
        total_wins = lambda df: df.groupby('driverId')['win']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def current_wins_inyear(df): # needs to be shifted because it needs to be until a certain race
    return df.assign(
        wins_this_year_cumulative = lambda df: df.groupby(['driverId','year'])['wins_this_year']
        .transform(lambda df: df.shift(1))
    )


@log_step
def get_total_poles(df):
    return df.assign(
        total_poles = lambda df: df.groupby('driverId')['pole']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def get_podiums(df):
    return df.assign(
        podium = lambda df: df['positionOrder'].isin([1,2,3])
    )


@log_step
def get_total_podiums(df):
    return df.assign(
        total_podiums = lambda df: df.groupby('driverId')['podium']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def get_podiums_per_circuit(df):
    return df.assign(
        podium_per_circuit = lambda df: df.groupby(['driverId','circuitId'])['podium']
        .transform(lambda df: df.shift(1).cumsum())
    )


@log_step
def change_datetime(df):
    return df.assign(
        dob = lambda df: pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce'),
        date = lambda df: pd.to_datetime(df['date'], format='%Y-%m-%d')
    )


@log_step
def get_driver_age(df):
    return df.assign(
        current_age_days = lambda df: df['date'] - df['dob'],
        current_age_years = lambda df: df['current_age_days']/np.timedelta64(1,'Y')
    )


@log_step
def get_career_years(df):
    return df.assign(
        career_years = lambda df: df.groupby('driverId')['year']
        .transform(lambda df: df - df.min())
    )


@log_step
def get_DNF(df):
    return df.assign(
        dnf = lambda df: ~df['statusId'].isin([1,11,12,13,14])
    )


@log_step
def get_DNF_last5races(df):
    return df.assign(
        dnf_last5 = lambda df: df.groupby('driverId')['dnf']
        .transform(lambda df: df.shift(1).rolling(5).sum())
    )


@log_step
def racecounter_per_driver(df):
    return df.assign(
        race_count = lambda df: df.groupby(['driverId'])['raceId'].cumcount()+1
    )


@log_step
def last_race(df):
    return df.assign(
        last_race = lambda df: df.groupby('year')['raceId'].transform(lambda df: df.max()) == df['raceId']
    )


@log_step
def current_championshipstanding(df):
    return df.assign(
        championship_standing_before_race = lambda df: df.groupby('driverId')['championship_standing'].shift(1)
    )


# @log_step
# def get_WC(df):
#     champion_dict = df[(df['last_race'] == True) & (df['championship_standing'] == 1)]['driverRef'].value_counts().to_dict()
#
#     return df.assign(
#         championships_won = lambda df: df.loc[:, 'driverRef'].map(champion_dict)
#     )


@log_step
def remove_remaining_columns(df):
    subset = [
        'rank',
        'points_1',
        'fastestLapTime',
        'fastestLapSpeed',
        'statusId',
        'dob',
        'nationality',
        'round',
        'status',
        'current_age_days',
        'number_x',
        'positionText_1',
        'laps',
        'time_x',
        'milliseconds',
        'fastestLap',
        'WC_points_thisyear',
        'championship_standing',
        'wins_this_year',
        'last_race'
    ]
    return df.drop(labels=subset, axis=1)


@log_step
def get_combinations(df):
    df1 = df.copy()
    return df.merge(df1, on='raceId', how='outer',suffixes=('_left', '_right'))


@log_step
def filter_combinations(df):
    return df[(df['driverId_left'] != df['driverId_right'])]


# @log_step
# def new_race(df):
#     return df.assign(
#         new_race = lambda df: (df['raceId'] != df['raceId'].shift(1)).astype(int)
#     )
#
#
# @log_step
# def get_total_races(df):
#     return df.assign(
#         total_races = lambda df: df['new_race'].cumsum()
#     )


@log_step
def get_winning_driver(df):
    return df.assign(
        leftwon = lambda df: (df['positionOrder_left'] < df['positionOrder_right']).astype(int)
    )


@log_step
def previous_race_duel(df):
    return df.assign(
        left_won_previous_race= lambda df: df.groupby(['driverId_left', 'driverId_right'])['leftwon']
        .transform(lambda df: df.shift(1))
    )


@log_step
def get_final_dataset(df):
    subset = [
        'raceId',
        'driverId_left',
        'constructorId_left',
        'grid_left',
        'win_left',
        'pole_left',
        'podium_left',
        'dnf_left',
        'position_left',
        'positionOrder_left',
        'driverRef_left',
        #'year_left',
        'circuitId_left',
        'name_left',
        'date_left',
        'driverId_right',
        'constructorId_right',
        'grid_right',
        'win_right',
        'pole_right',
        'podium_right',
        'dnf_right',
        'position_right',
        'positionOrder_right',
        'driverRef_right',
        'year_right',
        'circuitId_right',
        'name_right',
        'date_right']
    return df.drop(labels=subset, axis=1)


@log_step
def get_final_dataset_app(df):
    subset = [
        'raceId',
        'driverId_left',
        'constructorId_left',
        'grid_left',
        'win_left',
        'pole_left',
        'podium_left',
        'dnf_left',
        'position_left',
        'positionOrder_left',
        #'driverRef_left',
        #'year_left',
        'circuitId_left',
        #'name_left',
        'date_left',
        'driverId_right',
        'constructorId_right',
        'grid_right',
        'win_right',
        'pole_right',
        'podium_right',
        'dnf_right',
        'position_right',
        'positionOrder_right',
        #'driverRef_right',
        'year_right',
        'circuitId_right',
        'name_right',
        'date_right']
    return df.drop(labels=subset, axis=1)


@log_step
def fill_na_rows(df):
    return df.assign(
        win_per_circuit_left = lambda df: df['win_per_circuit_left'].fillna(0),
        poles_per_circuit_left = lambda df: df['poles_per_circuit_left'].fillna(0),
        total_wins_left = lambda df: df['total_wins_left'].fillna(0),
        total_poles_left = lambda df: df['total_poles_left'].fillna(0),
        total_podiums_left = lambda df: df['total_podiums_left'].fillna(0),
        podium_per_circuit_left = lambda df: df['podium_per_circuit_left'].fillna(0),
        dnf_last5_left = lambda df: df['dnf_last5_left'].fillna(0),
        championship_standing_before_race_left = lambda df: df['championship_standing_before_race_left'].fillna(0),
        wins_this_year_cumulative_left = lambda df: df['wins_this_year_cumulative_left'].fillna(0),
        win_per_circuit_right=lambda df: df['win_per_circuit_right'].fillna(0),
        poles_per_circuit_right=lambda df: df['poles_per_circuit_right'].fillna(0),
        total_wins_right=lambda df: df['total_wins_right'].fillna(0),
        total_poles_right=lambda df: df['total_poles_right'].fillna(0),
        total_podiums_right=lambda df: df['total_podiums_right'].fillna(0),
        podium_per_circuit_right=lambda df: df['podium_per_circuit_right'].fillna(0),
        dnf_last5_right=lambda df: df['dnf_last5_right'].fillna(0),
        championship_standing_before_race_right=lambda df: df['championship_standing_before_race_right'].fillna(0),
        wins_this_year_cumulative_right=lambda df: df['wins_this_year_cumulative_right'].fillna(0)

     )


@log_step
def drop_na_rows(df):
    subset = [
        'mean_position_till_date_left',
        'mean_position__percircuit_till_date_left',
        'result_previous_race_left',
        'mean_last5_races_left',
        'last_race_at_circuit_left',
        'mean_position_till_date_right',
        'mean_position__percircuit_till_date_right',
        'result_previous_race_right',
        'mean_last5_races_right',
        'last_race_at_circuit_right',
        'left_won_previous_race'
    ]
    return df.dropna(subset=subset)


def get_clean_df():
    return (read_data()
    .pipe(merge_constructors)
    .pipe(merge_drivers)
    .pipe(merge_races)
    .pipe(merge_status)
    .pipe(merge_driverstandings)
    .pipe(remove_columns)
    .pipe(rename_columns)
    .pipe(sort_races)
    .pipe(average_finishing)
    .pipe(average_finishing_percircuit)
    .pipe(result_previous_race)
    .pipe(mean_last_5races)
    .pipe(last_race_at_circuit)
    .pipe(get_wins)
    .pipe(get_wins_per_circuit)
    .pipe(get_poles)
    .pipe(get_poles_per_circuit)
    .pipe(get_total_wins)
    .pipe(current_wins_inyear)
    .pipe(get_total_poles)
    .pipe(get_podiums)
    .pipe(get_total_podiums)
    .pipe(get_podiums_per_circuit)
    .pipe(change_datetime)
    .pipe(get_driver_age)
    .pipe(get_career_years)
    .pipe(get_DNF)
    .pipe(get_DNF_last5races)
    .pipe(racecounter_per_driver)
    .pipe(last_race)
    .pipe(current_championshipstanding)
    .pipe(remove_remaining_columns)
    .pipe(get_combinations)
    .pipe(filter_combinations)
    .pipe(get_winning_driver)
    .pipe(previous_race_duel)
    .pipe(get_final_dataset)
    .pipe(fill_na_rows)
    .pipe(drop_na_rows)
    )


def get_clean_app_df():
    return (read_data()
            .pipe(merge_constructors)
            .pipe(merge_drivers)
            .pipe(merge_races)
            .pipe(merge_status)
            .pipe(merge_driverstandings)
            .pipe(remove_columns)
            .pipe(rename_columns)
            .pipe(sort_races)
            .pipe(average_finishing)
            .pipe(average_finishing_percircuit)
            .pipe(result_previous_race)
            .pipe(mean_last_5races)
            .pipe(last_race_at_circuit)
            .pipe(get_wins)
            .pipe(get_wins_per_circuit)
            .pipe(get_poles)
            .pipe(get_poles_per_circuit)
            .pipe(get_total_wins)
            .pipe(current_wins_inyear)
            .pipe(get_total_poles)
            .pipe(get_podiums)
            .pipe(get_total_podiums)
            .pipe(get_podiums_per_circuit)
            .pipe(change_datetime)
            .pipe(get_driver_age)
            .pipe(get_career_years)
            .pipe(get_DNF)
            .pipe(get_DNF_last5races)
            .pipe(racecounter_per_driver)
            .pipe(last_race)
            .pipe(current_championshipstanding)
            .pipe(remove_remaining_columns)
            .pipe(get_combinations)
            .pipe(filter_combinations)
            .pipe(get_winning_driver)
            .pipe(previous_race_duel)
            .pipe(get_final_dataset_app)
            .pipe(fill_na_rows)
            .pipe(drop_na_rows)
    )