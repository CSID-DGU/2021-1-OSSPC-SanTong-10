from django.http import HttpResponse

def index(request):
    # 모델 활용
    return HttpResponse("Hello, World. You're at the polls index.")

