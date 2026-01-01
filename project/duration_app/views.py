from django.shortcuts import render
import torch
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# Create your views here.
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync



async def async_send_broadcast_notification(message):
    channel_layer = get_channel_layer()
    await channel_layer.group_send(
        "notifications",  # The group name
        {
            "type": "send_notification",
            "message": message
        }
    )


# channel_layer = get_channel_layer() 26 10 2019
                # 23 11   10 1 2020 
# async_to_sync(channel_layer.group_send)(
#     "notifications",
#     {
#         "type": "send_notification",
#         "message": "New notification received!"
#     }
# )




# @csrf_exempt
# def send_notification(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         message = data.get("message", "Default message")

#         channel_layer = get_channel_layer()
#         async_to_sync(channel_layer.group_send)(
#             "notifications",
#             {
#                 "type": "send_notification",
#                 "message": message,
#             }
#         )
#         return JsonResponse({"status": "sent"})





import torch.nn as nn

def about(request):
    return render(request,"about-us.html")


def blog(request):
    return render(request,"blog.html")


def contact(request):
    return render(request,"contact.html")


def department(request):
    return render(request,"department.html")

    
def doctors(request):
    return render(request,"doctors.html")


    
def element(request):
    return render(request,"element.html")


    
def index(request):
    return render(request,"index.html")



    
def singleblog(request):
    return render(request,"single-blog.html")

def sign_in(request):
    return render(request,"sign_in.html")


def sign_up(request):
    return render(request,"sign_up.html")



def live_prediction_view(request):
    return render(request, "test.html")
