from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import json


def get_price_prediction(json_data):
    def create_district_array(district_name):
        """

            returns a np array of zeros with a one at index = district index

        """

        with open('districts.json') as f:
            dict = json.load(f)

        district_index = dict[district_name]

        try:
            before = np.zeros((district_index,))
            after = np.zeros((139 - district_index - 1,))  # not 140 because drop first = true
            final = np.append(before, 1)
            final = np.append(final, after)
            return final
        except ValueError:
            return np.append(1, np.zeros(139))

    def create_housing_type_array(type):
        if type == 'Comm Element Condo':
            return np.array([1, 0, 0, 0, 0])
        if type == 'Condo Apt':
            return np.array([0, 1, 0, 0, 0])
        if type == 'Condo Townhouse':
            return np.array([0, 0, 1, 0, 0])
        if type == 'Detached':
            return np.array([0, 0, 0, 1, 0])
        if type == 'Semi-Detached':
            return np.array([0, 0, 0, 0, 1])

    bathrooms = json_data['bathrooms']
    sqft = json_data['sqft']
    parking = json_data['parking']
    bedrooms_ag = json_data['bedrooms_ag']
    bedrooms_bg = json_data['bedrooms_bg']
    housing_type = json_data['housing_type']
    district = json_data['district']

    arr = np.array([bathrooms, sqft, parking, bedrooms_ag, bedrooms_bg])
    house = np.append(arr, create_housing_type_array(housing_type))
    house = np.append(house, create_district_array(district))

    model = load_model('toronto_housing_model_final.h5')
    scaler = joblib.load('house_scaler.pkl')

    house = scaler.transform([house])
    return int(model.predict(np.array(house))[0][0])


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route('/')
def hello_world():
    return render_template('index.html')


# model = load_model('toronto_housing_model_final.h5')
# scaler = joblib.load('house_scaler.pkl')
# apparently loading the model and scaler and passing them in gets ValueError. Wasted me 1hr

@app.route('/api/house', methods=['POST'])
def predict_house():
    data = request.data
    print(data)
    data = data.decode('utf8')
    price = get_price_prediction(json_data=json.loads(data))
    return jsonify(price)


@app.route('/api/test', methods=['POST'])
def return_data():
    return jsonify('success')


if __name__ == '__main__':
    app.run()
