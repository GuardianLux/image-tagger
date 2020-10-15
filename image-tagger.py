
from imageai.Detection import ObjectDetection
import os, argparse, json
from exif import Image

input_file = ""
path = os.getcwd()
tags = []

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("-i")
args = parser.parse_args()
input_file = args.i

# Image detection model setup
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# Get detections
detections = detector.detectObjectsFromImage(input_image=input_file, output_image_path=os.path.join(path , "imagenew.jpg"))
for eachObject in detections:
    tags.append(eachObject['name'])

# Get exif data
with open(input_file, 'rb') as image_file:
    image = Image(image_file)

image_data = {
    'name' : input_file,
    'tags' : tags,
    'camera make' : image.make,
    'camera model' : image.model,
    'aperture' : image.aperture_value,
    'exposure_time' : image.exposure_time,
    'focal_length' : image.focal_length,
    'focal_length_in_35mm' : image.focal_length_in_35mm_film,
    'shutter_speed' : image.shutter_speed_value,
    'metering_mode' : image.metering_mode,
    'white_balance' : image.white_balance
}

print(json.dumps(image_data))