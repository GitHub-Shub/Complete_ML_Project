from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                Weather=request.form.get('Weather'),
                Traffic_Level=request.form.get('Traffic_Level'),
                Time_of_Day=request.form.get('Time_of_Day'),
                Vehicle_Type=request.form.get('Vehicle_Type'),
                Preparation_Time_min=int(request.form.get('Preparation_Time_min')),
                
                Courier_Experience_yrs=float(request.form.get('Courier_Experience_yrs')),
                Distance_km=float(request.form.get('Distance_km'))

            )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
        

if __name__=="__main__":
    app.run(host="0.0.0.0",debug='True')        
