def get_duels_won(df):
    return df.assign(
        duels_won = lambda df: df.groupby('driverRef_left')[0]
        .transform(lambda df: df.sum())
    )