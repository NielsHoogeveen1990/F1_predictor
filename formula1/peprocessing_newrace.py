import pandas as pd

from formula1.preprocessing_newdata import read_data


PATH = 'new_data'
DRIVER_AMOUNT = 20


def read_driverstandings():
    names = ['driverStandingsId',
                 'raceId',
                 'driverId',
                 'points',
                 'position',
                 'positionText',
                 'wins']

    return pd.read_csv(f'{PATH}/driver_standings.csv', names = names)


def create_new_race(df):
    return df.tail(DRIVER_AMOUNT)


def create_new_index(df):

    last_index = df.index.max()
    new_index = last_index + 1

    index_list = [i for i in range(new_index, new_index + DRIVER_AMOUNT)]

    return df.set_index([pd.Index(index_list)])


def create_new_Idlist(df, col):

    last_Id = df[col].max()
    new_Id = last_Id + 1

    Id_list = [i for i in range(new_Id, new_Id + DRIVER_AMOUNT)]

    df[col] = Id_list

    return df


def create_new_Id(df, col):

    last_Id = df[col].max()
    new_Id = last_Id + 1

    df[col] = new_Id

    return df


def concat_results(new_df, old_df):
    return pd.concat([old_df, new_df])


# Create new dataframe for the results
def create_new_results():
    return (create_new_race(read_data())
            .pipe(create_new_index)
            .pipe(create_new_Idlist, col= 'resultId')
            .pipe(create_new_Id, col= 'raceId')
            .pipe(concat_results, read_data())
            )


# Create new dataframe for the driverStandings
def create_new_driverstandings():
    return (create_new_race(read_driverstandings())
            .pipe(create_new_index)
            .pipe(create_new_Idlist, col= 'driverStandingsId')
            .pipe(create_new_Id, col='raceId')
            .pipe(concat_results, read_driverstandings())
            )