from flask import Flask, render_template, request
from formula1.db import retrieve_prediction_data
import pickle

# application factory --> create_app

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

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

        df = retrieve_prediction_data(driver1, driver2, circuit, year)

        # drop the driver1, driver2, circuit and year and predict that one row dataframe with the model and return result
        result1 = df['driverRef_left'][0]
        result2 = df['driverRef_right'][0]

        return render_template('prediction.html', result1=result1.capitalize(), result2=result2.capitalize(), year= year, circuit = circuit)

    return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True,port=80)