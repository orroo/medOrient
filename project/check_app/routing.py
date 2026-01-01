from django.urls import re_path
from .consumer import *
 
websocket_urlpatterns = [
    re_path(r'ws/predict_bp/$', mat_PredictConsumer.as_asgi()),
]
