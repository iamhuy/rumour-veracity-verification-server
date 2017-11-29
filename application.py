'''
Simple Flask application to test deployment to Amazon Web Services
Uses Elastic Beanstalk and RDS

Author: Scott Rodkey - rodkeyscott@gmail.com

Step-by-step tutorial: https://medium.com/@rodkey/deploying-a-flask-application-on-aws-a72daba6bb80
'''
from __future__ import print_function
from features.build_features import collect_feature
from flask import Flask, render_template, request, make_response, jsonify
from models.predict import predict
import sys

# Elastic Beanstalk initalization
application = Flask(__name__)
application.debug=True
# change this to your own value
application.secret_key = 'cC1YCIWOj9GgWspgNEo2'   

@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        resp = make_response(render_template('index.html'), 200)
        return resp

    if request.method == 'POST':
        tweet = request.get_json()
        
    feature_vector = collect_feature(tweet)

    resp = make_response(jsonify(predict(feature_vector)), 200)

    return resp

if __name__ == '__main__':
    application.run(host='0.0.0.0')
