import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# initiating the flask app
app = Flask(__name__)
# reading the pickled regression model
model = pickle.load(open('model.pkl', 'rb'))


# redirecting to the html homepage
@app.route('/')
def home():
    return render_template('index.html')


# POST method for sending inofrmation to the site after processing
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    # using model's predict method on the created array from the input data
    prediction = model.predict(final_features)
    prediction = float(prediction)

    # categorizing teh AQI according to the Indian standards of AQI Indexing
    if prediction > 0 and prediction <= 50:
        prediction_2 = 'Good'
    elif prediction > 50 and prediction <= 100:
        prediction_2 = 'Satisfactory'
    elif prediction > 100 and prediction <= 200:
        prediction_2 = 'Moderate'
    elif prediction > 200 and prediction <= 300:
        prediction_2 = 'Poor'
    elif prediction > 300 and prediction <= 400:
        prediction_2 = 'Very Poor'
    elif prediction > 400:
        prediction_2 = 'Severe'
    else:
        prediction_2 = "Error"

    return render_template('index.html', prediction_aqi='The predicted AQI Index is : {}'.format(prediction), prediction_classifier='The predicted AQI Category: {}'.format(prediction_2))


if __name__ == "__main__":
    app.run(debug=True)
