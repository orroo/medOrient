from django.urls import re_path
from .consumer import *
 
websocket_urlpatterns = [
    re_path(r'ws/predict/$', PredictConsumer.as_asgi()),
    re_path(r'ws/notifications/$', NotificationConsumer.as_asgi()),
    re_path(r'ws/upload/$', UploadConsumer.as_asgi()),
]
