from django.db import models

class Attendance(models.Model):
    name = models.CharField(max_length=100)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.date} {self.time}"
