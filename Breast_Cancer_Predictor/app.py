from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pickle

app = Flask(__name__,template_folder='templates')

file = open('Final_cancer_model.pkl','rb')
ml_model = pickle.load(file)

@app.route("/")

def home():
    return render_template('cancer.html')

@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        print(request.form)
        concave_points_mean = float(request.form['concave points_mean'])
        area_mean = float(request.form['area_mean'])
        radius_mean = float(request.form['radius_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        pred_args = [concave_points_mean,area_mean,radius_mean,perimeter_mean,concavity_mean]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.reshape(1,-1)

        output = ml_model.predict(pred_args_arr)
        print(output)

        if(int(output)==1):
            prediction = "You have symptoms of getting Breast Cancer. Please consult the doctor"
        else:
            prediction = "You don't have any symptoms of Breast Cancer"

    return render_template("result.html",prediction_text=prediction)



if __name__ == "__main__":
    app.run(debug=True)
