from django.shortcuts import render, get_object_or_404



# about 첫 화면 출력
def index(request):
    return render(request, 'pybo/about.html')


# about page 출력
# def page(request, page_num):
    # context = {'page_num': page_num}
#     return render(request, 'pybo/about.html', context)
