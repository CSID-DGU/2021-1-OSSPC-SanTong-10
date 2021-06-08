from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

from .models import Test
from .models import GameRecords
import json



# return String
def index(request):
    # 모델 활용
    return HttpResponse("Hello, World. You're at the polls index.")

# Read JSON Object
def game_records(request):
    gr = GameRecords.objects.all()
    print(gr)


# Insert (* Important)
def insertTest(request):
    Test.objects.create(name="test")
    # return render(request, 'index.html')


# Post (JSON Object)
def testJson(request):
    if (request.method == 'POST') :
        jsonObject = json.loads(request.body)
        print(jsonObject)

        a = GameRecords.objects.get(id=792)
        print(a.review_list)
        a.review_list = ['7&7&99', '8&8&92']
        a.save()

    return JsonResponse({"result":"ok"}, status=200)


