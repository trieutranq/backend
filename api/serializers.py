from rest_framework import serializers
from .models import Image
# Create your models here.


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ('name', 'photo')
