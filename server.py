import requests
import re
from flask import Flask, render_template, request, send_from_directory
from flask_restful import Resource, Api
from flask_bootstrap import Bootstrap
from main import Classifier

app = Flask(__name__)
api = Api(app)
classifier = Classifier()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    words = request.args['words']
    if(words != ''):
        prediction = classifier.predict_label([words])
        return render_template('index.html', words=words, prediction=prediction[0])
    return render_template('index.html')


class Predict(Resource):
    def get(self):
        words = request.args['words']
        if(words != ''):
            prediction = classifier.predict_label([words])
            return {'prediction': prediction[0]}


api.add_resource(Predict, '/api/predict', endpoint='user')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=False)
