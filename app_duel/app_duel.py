from flask import Flask, render_template, request
from formula1.db import retrieve_prediction_data
import pickle

# application factory --> create_app

app = Flask(__name__)

model = pickle.load(open('retrained_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':

        driver1 = request.form['driver1']
        driver2 = request.form['driver2']
        circuit = request.form['circuit']
        year = request.form['year']

        drivers_list = [driver1, driver2]

        df = retrieve_prediction_data(driver1, driver2, circuit, year)

        # drop the driver1, driver2, circuit and year and predict that one row dataframe with the model and return result
        df.drop(['driverRef_left', 'name_left', 'driverRef_right', 'leftwon', 'year_left'], axis=1, inplace=True)

        if model.predict(df)[0] == 1:
            winner = drivers_list[0]
            loser = drivers_list[1]
        else:
            winner = drivers_list[1]
            loser = drivers_list[0]

        predict_prob = model.predict_proba(df)[0].max() * 100


        return render_template('prediction_test2.html',
                               winner=winner.capitalize(),
                               loser=loser.capitalize(),
                               year= year,
                               circuit = circuit,
                               predict_prob= round(predict_prob,2))

    return render_template('index.html')


# if __name__ == '__main__':
#     app_duel.run(debug=True,port=80)