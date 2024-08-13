import flask
import pandas as pd
import numpy as np
import pickle
import joblib
from flask import request, render_template, url_for
app = flask.Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["DEBUG"] = True
from flask_cors import CORS
CORS(app)
# main index page route
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load(open("xgb_MOODEL.pkl", "rb"))
    if request.method == "POST":
        sex = request.form.get("sex")
        age = request.form.get("age")
        current_systolic_bp = request.form.get("Current Systolic BP")
        current_diastolic_bp = request.form.get("Current diastolic BP")
        previous_systolic_bp = request.form.get("Previous Systolic BP")
        previous_diastolic_bp = request.form.get("Previous diastolic BP")        
        previous_diag_3 = request.form.get("Previous Diag 3")
        admitted = request.form.get("admitted")
        comorbidity_disease = request.form.get("Comorbidity Disease")
        stage_of_htn = request.form.get("Stage of HTN(LR)")
        drug_1 = request.form.get("Drug 1")
        drug_2 = request.form.get("Drug 2")
        drug_3 = request.form.get("Drug 3")

        final_arr = [sex, age, current_systolic_bp, current_diastolic_bp,
                    previous_systolic_bp, previous_diastolic_bp, previous_diag_3, admitted, comorbidity_disease, stage_of_htn,
                    drug_1, drug_2, drug_3]
    print("final_arr:", final_arr)
    # Drop the rows with null values
    data = np.array(final_arr)
    data = data.reshape(1, -1)
    data = np.nan_to_num(data)
 # Make the prediction
    prediction = model.predict(data)
    # Return the appropriate response
    if prediction[0] == 1:
        return render_template("index.html", prediction_text=  "hypertensive patient is Readmitted")
    else:
        return render_template("index.html", prediction_text=  "hypertensive patient is not Readmitted")
if __name__== "main":
    app.run(debug=False)
    
    