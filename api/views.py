# THU VIEN CHO API=================================================
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import base64
import cv2
from rest_framework.parsers import MultiPartParser, FormParser
# THU VIEN CHO MO HINH=============================================
from generator import Generator
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import numpy as np
# create the detector, using default weights
detector = MTCNN()
# extract a single face from a given photograph


def extract_face(image, required_size=(256, 256)):
    image = np.array(image)
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


generator = Generator(HEIGHT=256, WIDTH=256)
generator.load_weights('c_generator.h5')
print('Loaded')
# =================================================================
# Create your views here.
# python ./manage.py runserver 0.0.0.0:8000


class ImageAPI(APIView):
    parser_classes = (MultiPartParser, FormParser, )

    def get(self, request):
        return Response(status=status.HTTP_200_OK)

    def post(self, request):
        image_base64 = str(request.data['base64'])
        image_uri = str(request.data['uri'])
        name = image_uri.split('/')[-1]
        with open("images/"+name, "wb") as fh:
            fh.write(base64.b64decode(image_base64))

        return Response(status=status.HTTP_200_OK)
