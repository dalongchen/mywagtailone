from django.urls import path
from . import views

app_name = 'datatables'
urlpatterns = [
    path('', views.index, name='index'),
    path('index_django/', views.index_django, name='index_django'),
    path('search_integration/', views.search_integration, name='search_integration'),
    path('east_money_lgt/', views.east_money_lgt, name='east_money_lgt'),
    path('stock_details/', views.stock_details, name='stock_details'),
    path('east_data/', views.east_data, name='east_data'),
]
