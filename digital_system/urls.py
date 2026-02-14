from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Home page
    path('', views.home, name='home'),

    # Attendance page (camera + result)
    path('attendance/', views.attendance_register, name='attendance'),

    # Staff login
    path('staff-login/', views.staff_login, name='staff_login'),

    # Live camera stream for face detection
    path('camera/', views.start_face_detection, name='start_face_detection'),

]
