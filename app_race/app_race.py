from flask import Flask, render_template, request
import pandas as pd
from formula1.db import retrieve_prediction_data_race
from app_race.dataprocessesing import get_duels_won
import pickle

# application factory --> create_app

app = Flask(__name__)

model = pickle.load(open('trained_models/model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':

        circuit = request.form['circuit']
        year = request.form['year']

        df = retrieve_prediction_data_race(circuit, year)

        df_predict = df.drop(['driverRef_left', 'name_left', 'driverRef_right', 'leftwon', 'year_left'], axis=1)

        # Make prediction
        prediction = model.predict(df_predict)

        # Convert prediction to Series
        prediction = pd.Series(prediction)

        df = df.merge(prediction.to_frame(), left_index=True, right_index=True)

        df_prediction = get_duels_won(df)

        race_result = (df_prediction.groupby('driverRef_left')[['duels_won', 'driverRef_left']]
                       .max()
                       .sort_values(ascending=False,
                                    by='duels_won'))

        first = race_result['driverRef_left'][0]
        second = race_result['driverRef_left'][1]
        third = race_result['driverRef_left'][2]
        fourth = race_result['driverRef_left'][3]
        fifth = race_result['driverRef_left'][4]
        sixth = race_result['driverRef_left'][5]
        seventh = race_result['driverRef_left'][6]
        eighth = race_result['driverRef_left'][7]
        ninth = race_result['driverRef_left'][8]
        tenth = race_result['driverRef_left'][9]

        return render_template('prediction_test2.html',
                               first= first.capitalize(),
                               second= second.capitalize(),
                               third= third.capitalize(),
                               fourth= fourth.capitalize(),
                               fifth= fifth.capitalize(),
                               sixth= sixth.capitalize(),
                               seventh=seventh.capitalize(),
                               eighth=eighth.capitalize(),
                               ninth=ninth.capitalize(),
                               tenth=tenth.capitalize(),
                               year= year,
                               circuit = circuit)

    return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True,port=80)