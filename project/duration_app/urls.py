from django.contrib import admin
from django.urls import path
from .views import *


urlpatterns = [
    path('about/', about,name='about'),
    path('blog/', blog,name='blog'),
    path('contact/', contact,name='contact'),
    path('department/', department,name='department'),
    path('doctors/', doctors,name='doctors'),
    path('element/', element,name='element'),
    path('index/', index,name='index'),
    path('singleblog/', singleblog, name='singleblog'),
    # Chatbot endpoints (match views)
    path('chatbot/sump/', sump_chat_bot, name='sump_chatbot'),
    path('chatbot/cnam/', cnam_chat_bot, name='cnam_chatbot'),
    path('chatbot/vac/', vac_chat_bot, name='vac_chatbot'),
    path('chatbot/ord/', ord_chat_bot, name='ord_chatbot'),
    path('chatbot/pdf/', pdf_chat_bot , name='pdf_chatbot'),
    # User auth pages
    path('sign-in/', sign_in, name='sign_in'),
    path('sign-up/', sign_up, name='sign_up'),
    # Prediction endpoints
    path('lung_predict/', lung_predict, name='lung_predict'),
    path('carrie_predict/', carrie_predict, name='carrie_predict'),
    # path('live/', live_prediction_view, name='live_prediction'),

    
    # path('pred/', F_D_pred,name='pred'),
    # path('live/', live_prediction_view, name='live_prediction'),
    # path('send-notification/', send_notification, name='send_notification'),
]

