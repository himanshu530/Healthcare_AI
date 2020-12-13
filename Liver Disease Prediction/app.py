from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")

file = open('Final_liver_model.pkl','rb')
ml_model = pickle.load(file)

@app.route("/")

def home():
    return render_template("liver.html")

@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":
        print(request.form)
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float,to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1,-1)
        output = ml_model.predict(to_predict)
        print(output)
        if (int(output)==1):
            prediction = "You have Symptoms of getting Liver Disease. Please consult the Doctor"
        else:
            prediction = "You dont have any Symptoms of the Liver Disease"
    return render_template("predict.html",prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)
