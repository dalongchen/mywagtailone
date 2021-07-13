import json
from django.http import HttpResponse
from django.shortcuts import render
from mywagtailone.home.models import ShgtDf2021


def home(request):  # 数据展示
    return render(request, 'home_page.html')


def ShgtDf2021(request):  # ajax的url
    data_list = []
    for data_info in ShgtDf2021.objects.all():
        data_list.append({
            'code': data_info.code,
            'name': data_info.name,
            'todaycloseprice': data_info.todaycloseprice,
            'todayup': data_info.todayup,
            'todayquantity': data_info.todayquantity,
            'todayvalue': data_info.todayvalue,
            'circulaterate': data_info.circulaterate,
            'totalrate': data_info.totalrate,
            'addnumber': data_info.addnumber,
            'addvalue': data_info.addvalue,
            'addvaluerate': data_info.addvaluerate,
            'addvalueratecirculate': data_info.addvalueratecirculate,
            'addvalueratetotal': data_info.addvalueratetotal,
            'trade': data_info.trade,
            'date': data_info.date,
        })

    data_dic = {}
    data_dic['data'] = data_list  # 格式一定要符合官网的json格式，否则会出现一系列错误
    return HttpResponse(json.dumps(data_dic))