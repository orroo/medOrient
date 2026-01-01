import json
from channels.generic.websocket import AsyncWebsocketConsumer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime , date

from channels.layers import get_channel_layer
import torch.nn as nn
# from .views import F_D_model, F_D_scalerX, F_D_scalerY  # Adjust import path
from asgiref.sync import async_to_sync ,asyncio
import requests
import joblib
import torch
from .views import *



live_url = 'http://127.0.0.1:5000/mat_data'

class mat_PredictConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({"message": "Connected to prediction server."}))

    async def receive(self,text_data):

        channel_layer = get_channel_layer()
        
        # # Broadcast to group, excluding a specific user (if needed)
        # # async_to_sync(
        # await channel_layer.group_send(
        #     "notifications",  # Group name
        #     {
        #         "type": "send_notification",
        #         "message": 'test call:D',
        #     }
        # )

        try:
    
            response = requests.get(live_url)
    
            if response.status_code == 200:
                X = np.array(response.json())  # This is your random sample as a Python dict
                print(X)
            else:
                print(f"Error: {response.status_code}")

            X = F_D_scalerX.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X = torch.from_numpy(X).float()
            predictions = F_D_model(X)
            output = F_D_scalerY.inverse_transform(predictions.detach().numpy())
            print(output)
            print(output[0][0])
            print(output[0][1])
            if ( output[0][0] > 180 and output[0][1] > 120 ):
                await channel_layer.group_send(
                    "notifications",  # Group name
                    {
                        "type": "send_notification",
                        "message": 'Patient Detected with Critical Hypertension',
                    }
                )
            elif  ( output[0][0] > 130 and output[0][1] > 90 ):
                await channel_layer.group_send(
                    "notifications",  # Group name
                    {
                        "type": "send_notification",
                        "message": 'Patient Detected with Hypertension',
                    }
                )


            # Pass results to template
            await self.send(text_data=json.dumps({
                'prediction_sdp': float(output[0][0]),
                'prediction_dbp': float(output[0][1]),
            }))
        except Exception as e:
            
            # print( "in not 2")
            await self.send(text_data=json.dumps({"error": str(e)}))

    async def disconnect(self, close_code):
        pass
