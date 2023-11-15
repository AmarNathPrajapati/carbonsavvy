from flask import Flask,render_template,request,session,redirect,flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os
from werkzeug.utils import secure_filename
import math
import pickle
import numpy as np
from joblib import dump,load
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

carbon_emission = pd.read_csv('data.csv')

from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(carbon_emission,test_size=0.2,random_state=42)

# # Stratify Sampling

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(carbon_emission,carbon_emission['Garbage']):
    stratify_train_set = carbon_emission.loc[train_index]
    stratify_test_set = carbon_emission.loc[test_index]

carbon_emission = stratify_train_set.copy()

# # Handling Missing Values
from sklearn.impute import SimpleImputer

co2_factors = stratify_train_set.drop("CO2_Emission",axis=1)
co2_emission_labels = stratify_train_set["CO2_Emission"].copy()


# # Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])
carbon_emission_tr = my_pipeline.fit_transform(co2_factors)

app = Flask(__name__)
model = pickle.load(open('carbonSavvy.pkl','rb'))

with open('static/config.json','r') as c:
    params= json.load(c)["params"]
    
# connection with database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/carbonsavvy'
db = SQLAlchemy(app)
app.secret_key = 'super-secret-key'

class Prediction(db.Model):
    # CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV

    sno = db.Column(db.Integer, primary_key=True)
    wind = db.Column(db.String(80), nullable=False)
    vehicle = db.Column(db.String(120),nullable=False)
    solar=db.Column(db.String(120), nullable=False)
    garbage = db.Column(db.String(120),nullable=False)
    evehicle = db.Column(db.String(120), nullable=False)
    industries=db.Column(db.String(120), nullable=False)
    forest= db.Column(db.String(150), nullable=False)
    population = db.Column(db.String(80), nullable=False)
    bio = db.Column(db.String(120),nullable=False)
    recycle=db.Column(db.String(120), nullable=False)
    thermal = db.Column(db.String(120), nullable=False)
    buildings=db.Column(db.String(120), nullable=False)
    nuclear= db.Column(db.String(150), nullable=False)
    co2 = db.Column(db.String(80), nullable=False)
    

@app.route("/")
def home():
    return render_template('dashboard.html',params=params)

 
@app.route('/predict_value')
def predict_data():
    return render_template('predict_value.html',params=params)

@app.route('/predictions')
def predictions():
    predictions = Prediction.query.all();
    return render_template('predictions.html',predictions=predictions,params=params)


@app.route('/predict', methods=['POST'])
def predict():
    # CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV

    data1 = request.form.get('a',type=float)
    data2 = request.form.get('b',type=float)
    data3 = request.form.get('c',type=float)
    data4 = request.form.get('d',type=float)
    data5 = request.form.get('e',type=float)
    data6 = request.form.get('f',type=float)
    data7 = request.form.get('g',type=float)
    data8 = request.form.get('h',type=float)
    data9 = request.form.get('i',type=float)
    data10 = request.form.get('j',type=float)
    data11 = request.form.get('k',type=float)
    data12 = request.form.get('l',type=float)
    data13 = request.form.get('m',type=float)
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    prepared_data = my_pipeline.transform(arr)
    pred = model.predict(prepared_data)
    pred = round(pred[0],3)
    entry = Prediction(wind=data1,vehicle=data2,solar=data3,garbage=data4,evehicle=data5,industries=data6,forest=data7,population=data8,bio=data9,recycle=data10,thermal=data11,buildings=data12,nuclear=data13,co2=pred);
    db.session.add(entry)
    db.session.commit()
    return render_template('after.html',data=pred,params=params)

app.run(debug=True)