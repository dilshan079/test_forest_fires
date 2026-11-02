import pickle
from pathlib import Path
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regession model and standard scaler


ridge_model=pickle.load(open("models/ridge_model.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))



    


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            fields = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]
            values = {}
            for field in fields:
                raw_value = request.form.get(field, "").strip()
                if field == "Classes":
                    class_value = raw_value.lower()
                    if class_value in {"fire", "1"}:
                        values[field] = 1.0
                    elif class_value in {"not fire", "0"}:
                        values[field] = 0.0
                    else:
                        raise ValueError("Classes must be fire, not fire, 1, or 0")
                else:
                    values[field] = float(raw_value)

            input_data = pd.DataFrame([[values[field] for field in fields]], columns=fields)
            scaled_data = standard_scaler.transform(input_data)
            result = ridge_model.predict(scaled_data)
            return render_template("home.html", results=float(result[0]))
        except (TypeError, ValueError) as error:
            return render_template("home.html", error=f"Invalid input: {error}"), 400

    return render_template("home.html")


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5007)

