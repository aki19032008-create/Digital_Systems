from django.db import models

class Student(models.Model):
    roll = models.CharField(max_length=20)
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='students/')

    def __str__(self):
        return self.name


class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    status = models.CharField(max_length=10, default="Present")

