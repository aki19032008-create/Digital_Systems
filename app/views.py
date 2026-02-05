from django.shortcuts import render

def home(request):
    return render(request, 'app/home.html')

def attendance_register(request):
    return render(request, 'app/attendance_register.html')

def staff_login(request):
    return render(request, 'app/staff_login.html')

