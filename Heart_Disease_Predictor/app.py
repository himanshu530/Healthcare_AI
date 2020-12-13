from flask import Flask, render_template, flash,redirect
import joblib
from flask import request
import numpy as np
import pickle

app = Flask(__name__)

file = open('final_model.pkl','rb')
ml_model = pickle.load(file)

@app.route('/')

def home():
    return render_template('heart.html')

@app.route("/predict",methods = ["POST"])
def predict():
    if request.method == "POST":
        print(request.form)
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        pred_args = [cp,trestbps,chol,fbs,restecg,thalach,exang]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.reshape(1,-1)
        output = ml_model.predict(pred_args_arr)
        print(output)

        if (int(output)==1):
            prediction = "You have Symptoms of getting heart disease. Please consult the Doctor"
        else:
            prediction = "You dont have any Symptoms of the heart disease"

    return render_template("result.html",prediction_text=prediction)



if __name__ == "__main__":
    app.run(debug=True)
