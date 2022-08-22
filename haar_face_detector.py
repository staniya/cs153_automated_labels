# import the necessary packages
import argparse
import imutils
import cv2
from pathlib import Path
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
detector = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
# load the input image from disk, resize it, and convert it to
# grayscale
image_name = args["image"]
image = cv2.imread(image_name)
image = imutils.resize(image, width=500)
film_name = Path(image_name).parts[1]
image_name = Path(image_name).parts[2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the input image using the haar cascade face
# detector

rects = detector.detectMultiScale(gray, scaleFactor=1.05,
                                  minNeighbors=7, minSize=(30, 30),
                                  flags=cv2.CASCADE_SCALE_IMAGE)
# loop over the bounding boxes
for (x, y, w, h) in rects:
    # draw the face bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show the output image
cv2.imwrite(f"./open_cv_figures/{film_name}/{image_name}", image)
