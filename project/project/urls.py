"""
URL configuration for project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path , include
from duration_app import views
from django.conf import settings            
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.sign_in, name='sign_in'),
    path('home/', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('department/', views.department, name='department'),
    path('doctors/', views.doctors, name='doctors'),
    path('blog/', views.blog, name='blog'),
    path('singleblog/', views.singleblog, name='singleblog'),
    path('contact/', views.contact, name='contact'),
    path('element/', views.element, name='element'),
    path('sign_in/', views.sign_in, name='sign_in'),
    path('sign_up/', views.sign_up, name='sign_up'),
    
    path('lung_predict/', views.lung_predict,name='lung_predict'),
    path('carrie_predict/', views.carrie_predict,name='carrie_predict'),
    path('stroke/', include('stroke_detection_app.urls')),


    path('chatbot/sump/', views.sump_chat_bot, name='sump_chatbot'),
    path('chatbot/cnam/', views.cnam_chat_bot, name='cnam_chatbot'),
    path('chatbot/vac/', views.vac_chat_bot, name='vac_chatbot'),
    path('chatbot/ord/', views.ord_chat_bot, name='ord_chatbot'),
    path('chatbot/pdf/', views.pdf_chat_bot , name='pdf_chatbot'),


 
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    
    
 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
