import json
from channels.generic.websocket import AsyncWebsocketConsumer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime , date

from channels.layers import get_channel_layer
import torch
# from .views import F_D_model, F_D_scalerX, F_D_scalerY  # Adjust import path
from asgiref.sync import async_to_sync ,asyncio




import requests

# URL of the Flask endpoint
upload_url = 'http://127.0.0.1:5000/random_sample'

predict_url = 'http://127.0.0.1:5000/predict'

# # Send GET request
# response = requests.get(upload_url)

# # Check response status
# if response.status_code == 200:
#     data = response.json()  # This is your random sample as a Python dict
#     print(data)
# else:
#     print(f"Error: {response.status_code}")





# class NotificationConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.group_name = "notifications"
#         await self.channel_layer.group_add(self.group_name, self.channel_name)
#         await self.accept()

#     async def disconnect(self, close_code):
#         await self.channel_layer.group_discard(self.group_name, self.channel_name)

#     async def receive(self, text_data):
#         print(text_data)
#         data = json.loads(text_data)
#         # self.send_notification(data.get("message"))
#         await self.send(text_data=json.dumps({
#             "message": data.get("message")
#         }))
#         pass

#     async def send_notification(self, event):
#         await self.send(text_data=json.dumps({
#             "message": event["message"]
#         }))
# /*************  ✨ Windsurf Command ⭐  *************/
class UploadConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "upload-data"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        # Do something with the data
        
        # sample = pd.read_csv('duration_app\pkl\samples_of_a_patient.csv')
        # sample.sample(1).to_json(orient='records')
        sent = 0
        while sent==0:
            response = requests.get(upload_url)

            # Check response status
            if response.status_code == 200:
                data = response.json()  # This is your random sample as a Python dict
                print(data)
            else:
                print(f"Error: {response.status_code}")
            # data = json.loads(sample.sample(1).to_json(orient='records'))
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "send_data",
                    "message": data
                }
            )
            await asyncio.sleep(3)
            sent = 1
# /*************  ✨ Windsurf Command ⭐  *************/
    async def send_data(self, event):
        await self.send(text_data=json.dumps({
            "message": event["message"]
        }))
# /*******  9af41f01-9e47-43ff-961c-de3c7f824073  *******/
        

class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "notifications"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message = data.get("message")
            sender_channel = self.channel_name
            # Broadcast the message to all connected clients
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "send_notification",
                    "message": message,
                    "sender_channel": sender_channel 
                }
            )
        except json.JSONDecodeError:
            # Handle invalid JSON
            await self.send(text_data=json.dumps({
                "error": "Invalid JSON format"
            }))

        

    async def send_notification(self, event):
        try :
            if event["sender_channel"] : 
                if self.channel_name != event["sender_channel"]:
                    await self.send(text_data=json.dumps({
                        "message": event["message"]
                    }))
        except KeyError: 
                await self.send(text_data=json.dumps({
                    "message": event["message"]
                }))


class PredictConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({"message": "Connected to prediction server."}))

    async def receive(self, text_data):

        channel_layer = get_channel_layer()
        
        # Broadcast to group, excluding a specific user (if needed)
        # async_to_sync(
        # await channel_layer.group_send(
        #     "notifications",  # Group name
        #     {
        #         "type": "send_notification",
        #         "message": 'test call:D',
        #     }
        # )



        data = json.loads(text_data)
        input_features = data.get("input")
        if input_features is None:
            await self.send(text_data=json.dumps({"error": "No input data provided."}))
            return

        print(input_features)
    #     treatment = pd.DataFrame([input_features])

    #     treatment["dialysisstart"] = pd.to_datetime(treatment["dialysisstart"])

    #     treatment["Ds_hour"] = pd.to_datetime( treatment.dialysisstart).dt.hour
    #     treatment["Ds_minutes"] = pd.to_datetime( treatment.dialysisstart).dt.minute

        
    #     treatment["keyindate"] = pd.to_datetime(treatment["keyindate"])
    #     treatment["session_year"] =  pd.to_datetime( treatment.keyindate).dt.year
    #     treatment["session_month"] = pd.to_datetime( treatment.keyindate).dt.month
    #     treatment["session_dayofweek"] = pd.to_datetime( treatment.keyindate).dt.dayofweek
    #     treatment["session_is_weekend"] = treatment["session_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

        
    #     treatment["first_dialysis"] = pd.to_datetime(treatment["first_dialysis"])
    #     treatment["fd_year"] =  pd.to_datetime( treatment.first_dialysis).dt.year
    #     treatment["fd_month"] = pd.to_datetime( treatment.first_dialysis).dt.month

    #     datatime = datetime.now()
        
    #     treatment["datatime"] = pd.to_datetime(datatime)

    #     treatment["year"] =  pd.to_datetime(treatment["datatime"]).dt.year
    #     treatment["month"] = pd.to_datetime( treatment["datatime"]).dt.month
    #     treatment["dayofweek"] = pd.to_datetime( treatment["datatime"]).dt.dayofweek
    #     treatment["is_weekend"] = treatment["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    #     treatment["hour"] = pd.to_datetime( treatment["datatime"]).dt.hour
    #     treatment["minutes"] = pd.to_datetime( treatment["datatime"]).dt.minute
        

    #     # ordinal = OrdinalEncoder(categories=[['SI', 'NO']])
    #     # treatment.hypotension = ordinal.fit_transform(treatment.hypotension.values.reshape(-1, 1))
    #     print( treatment.shape)


    #     X= treatment[['sbp', 'dbp', 'temperature', 'conductivity', 'uf',
    #    'blood_flow', 'gender', 'birthday', 'DM',
    #     'weightstart', 'weightend', 'dryweight',
    #    'pulse', 'respiratory_rate', 'blood_oxygen_lvl', 'glucose_lvl',
    #    'hypotension', 'age', 'dialyzer', 'bath', 'technique', 'gain',
    #    'bath_temperature', 'replacement_Volume', 'kt', 'Bath_Flow',
    #    'bicarbonate_conductivity', 'arterial_Pressure', 'Venous_Pressure',
    #    'transmembrane_Pressure', 
    #     'Ds_hour', 'Ds_minutes', 'session_year',
    #    'session_month', 'session_dayofweek', 'session_is_weekend', 'year',
    #    'month', 'dayofweek', 'is_weekend', 'hour', 'minutes', 'fd_year',
    #    'fd_month']]

        try:
    #         print( X.shape)
    #         print( X.columns)
    #         # if  X is None or len(X) != 44:
                
    #         #     print( "in not 1")
    #         #     raise ValueError("Invalid input")

    #         # input_df = pd.DataFrame([input_features])
            
    #         # print( "in2")
    #         duration, freq = predict(F_D_model, X)
            response = requests.post(predict_url, json=input_features)

            if response.status_code != 400:
                response_data = response.json()
                duration = response_data.get("duration")
                freq = response_data.get("frequency")
            else:
                duration = None
                freq = None
            print( duration)
            print( freq)
            await self.send(text_data=json.dumps({
                "duration": float(duration),
                "frequency": float(freq)
            }))
        except Exception as e:
            
            # print( "in not 2")
            await self.send(text_data=json.dumps({"error": str(e)}))

    async def disconnect(self, close_code):
        pass
