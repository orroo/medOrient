 
from django.urls import path
from .views import face,start_camera
urlpatterns = [
path('recognition/',face,name='face'),
path('camera/',start_camera, name='start_camera'),]

 