import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,Weather:str,Traffic_Level:str,Time_of_Day:str,Vehicle_Type:str,Preparation_Time_min:int,Courier_Experience_yrs:int,Distance_km:int):
        self.Weather=Weather
        self.Traffic_Level=Traffic_Level
        self.Time_of_Day=Time_of_Day
        self.Vehicle_Type=Vehicle_Type
        self.Preparation_Time_min=Preparation_Time_min
        self.Courier_Experience_yrs=Courier_Experience_yrs
        self.Distance_km=Distance_km
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Weather": [self.Weather],
                "Traffic_Level": [self.Traffic_Level],
                "Time_of_Day": [self.Time_of_Day],
                "Vehicle_Type": [self.Vehicle_Type],
                "Preparation_Time_min": [self.Preparation_Time_min],
                "Courier_Experience_yrs": [self.Courier_Experience_yrs],
                "Distance_km": [self.Distance_km],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
