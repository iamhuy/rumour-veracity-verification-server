'''
Simple Flask application to test deployment to Amazon Web Services
Uses Elastic Beanstalk and RDS

Author: Scott Rodkey - rodkeyscott@gmail.com

Step-by-step tutorial: https://medium.com/@rodkey/deploying-a-flask-application-on-aws-a72daba6bb80
'''
from __future__ import print_function

from flask import Flask, render_template, request, make_response
import sys

# Elastic Beanstalk initalization
application = Flask(__name__)
application.debug=True
# change this to your own value
application.secret_key = 'cC1YCIWOj9GgWspgNEo2'   

@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():
    resp = make_response(render_template('index.html'), 200)

    if request.method == 'GET':
        return resp

    if request.method == 'POST':
        print(request.get_json(), sys.stderr)

    return resp

if __name__ == '__main__':
    application.run(host='0.0.0.0')
