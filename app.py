import  pickle
import  numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

lin_reg = pickle.load(open('models/lin_reg.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_sc=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=lin_reg.predict(new_data_sc)

        return render_template('index.html', result=result[0])
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)

def home():
    return render_template('index.html')