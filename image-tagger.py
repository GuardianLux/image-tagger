
from imageai.Detection import ObjectDetection
import os, argparse
from exif import Image

input_file = ""
path = os.getcwd()

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
    print(eachObject["name"] , " : " , eachObject["percentage_probability"])

# Get exif data
with open(input_file, 'rb') as image_file:
    image = Image(image_file)

print(dir(image))
