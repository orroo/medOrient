# corrected leakage_detection_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('',         views.leak_form,    name='leak_form'),
    path('predict/', views.leak_predict, name='leak_predict'),
]
