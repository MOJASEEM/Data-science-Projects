from flask import Flask, request, jsonify, render_template
import util
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Moves up one level (..) then into the 'client' directory
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'client')
app = Flask(__name__, template_folder=TEMPLATE_DIR)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.route('/')
def home():
    # If you have an index.html in templates/
    return render_template('app.html')
@app.route('/favicon.ico')
def favicon():
    return "", 200
if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()
