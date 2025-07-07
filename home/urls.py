from django.urls import path
from . import views
from .views import predict_aqi

app_name = 'home'

urlpatterns = [
    path('', views.index, name='index'),                  # Homepage
    path('blog/', views.blog, name='blog'),               # Blog page
    path('about/', views.about, name='about'),            # About page
    path('contact/', views.contact, name='contact'),      # Contact page
    path('predict/', predict_aqi, name='predict_aqi'),    # AQI prediction
]
