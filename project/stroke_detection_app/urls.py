from django.urls import path
from . import views

app_name = 'stroke_detection_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('status/', views.status, name='status'),
    path('run/', views.start_detection, name='start_detection'),
    path('upload/', views.upload, name='upload'),
    path('live/', views.live, name='live'),
    path('preload/', views.preload, name='preload'),
    path('debug_face/', views.debug_face, name='debug_face'),
]
