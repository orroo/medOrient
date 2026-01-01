from django.urls import path
from .views import Kidney,predict_view
urlpatterns = [
path('',Kidney,name='kidney'),
path('predict',predict_view, name='predect'), ]