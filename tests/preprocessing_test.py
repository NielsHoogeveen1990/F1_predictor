from formula1.db import retrieve_prediction_data_race
import pytest

def test_last_race_at_circuit():
    df = retrieve_prediction_data_race('Spanish Grand Prix', 2019)

    assert df[df['driverRef_left'] == 'vettel']['last_race_at_circuit_left'].values[0] == 4
    assert df[df['driverRef_left'] == 'hamilton']['last_race_at_circuit_left'].values[0] == 1


