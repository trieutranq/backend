from django.db import models
from rest_framework import serializers
# Create your models here.


class Image(models.Model):
    name = models.CharField(
        max_length=255,
    )
    photo = models.ImageField(
        max_length=None,
        upload_to='images',
    )
