# Using Deep Learning Techniques For Face Detection On Cinematic Eye-Tracking Data

## Introduction

As interest in the interplay between artificial intelligence and human-computer interaction increases, researchers attempt to define models that accurately describe novel interactions between humans and machines. Amongst the many interactions that researchers have been studying, eye-tracking has gained increasing attention. This is because many studies have found that facial biometrics have more distinct features that allow for them to have advantages over other biometrics such as palmistry and fingerprint. Since eye movement is the most frequent of all facial movements, it is logical that eye-tracking research has garnered increasing attention. Furthermore, since eye movements are fundamental to the operation of the visual system, the movement of the user’s eyes can provide a convenient and natural source of data input. Thus, eye-tracking technology is applicable in many domains such as psychology, marketing, medical, computer gaming, and cognitive science. Hence, there is a need to optimize eye-tracking algorithms so that they can be widely adopted for classification tasks.

In eye-tracking research, a fundamental process is to perform feature labeling so that eye-tracking data can be categorized based on the essential ocular activity indicators. Traditionally, feature labeling was largely done by humans which were both tedious and inefficient. Consequently, to help researchers minimize the work and effort of data labeling from scratch, we present machine learning models that perform automated feature labeling, specifically a bounding box prediction and classification of human heads, on the [Gaze Data for the Analysis of Attention in Feature Films](https://graphics.stanford.edu/~kbreeden/gazedata.html). The dataset contains robust gaze information of human participants in response to carefully curated film clips and has been augmented via the annotation of selected high-level cinematographic and ocular activity features.

In this work, we employ several architectures for face detection to assist contemporary research done to study the allocation of human gaze over visual stimuli. The study will involve three parts. The first is a baseline model that uses Haar Cascade, an object detection algorithm used to identify features of a face in an image or a real-time video. This approach performs feature detection on an image containing a face, based on the ratio of the intensities of the pixels within the target image. The second is to apply Vora et al.'s Fully Conventional Head Detector (FCHD), an end-to-end trainable head detection model, to perform classification and implement a localization function for the Gaze Data. The third is to employ Deepface, a face recognition and facial attribute analysis framework that wraps state-of-the-art models including [VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [Google FaceNet](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [Google OpenFace](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [Facebook DeepFace](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [DeepID](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [ArcFace](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), and [Dlib](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). By exploring and implementing these state-of-the-art models, this study seeks to identify the limitations of current algorithms in detecting people under diverse camera conditions, human poses, and lighting.

## Research Paper:

The research paper is included within this GitHub Repository titled: _Taniya_Shinn_Report_. Click [here](https://github.com/staniya/cs153_automated_labels/blob/main/Taniya_Shinn_Report.pdf) to access the PDF.

## Repository Structure

```
.
├── face-detection.ipynb <- Primary File For Face Detection On Cinematic Eye-Tracking Data
├── Taniya_Shinn_Report.pdf <- Research Paper
├── .gitignore
├── README.md
├── video-detector.py <- Face Detection Within Video Stream
└── haar_face_detector.py <- Face Detection with Haar Cascade

```

## Methods

Given that the Haar cascade model is not a deep learning face detector, the latter models should have significantly higher accuracy and more robust face detections. However, the benefit of using Haar cascades is that they are computationally efficient, have low memory requirements, and have a small model size.

The development environment requires OpenCV and [haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml), the pre-trained face detector provided by the developers and maintainers of the OpenCV library. The _cv2.CascadeClassifier()_ is used to load the detector on disk. Then, taking the grayscale images generated from the data preprocessing phase, we use _cv2.rectangle()_ to draw a bounding box around the detected faces. Note that the bounds of the boxes are determined by the classifier’s _detectMultiScale()_ function which detects the faces in the input and returns a list of bounding boxes (represented using x and y coordinates where the faces are in the image).

To discuss our choice of parameters, the _scaleFactor_ controls the extent to which the image size is reduced at each scale. In this experiment, we selected a value of 1.05 so that the size of the image is reduced by 5\% at each level in the [scale pyramid](https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/). Next, _minNeighbors_ parameter controls the number of neighbors each window should have for the area in the window to be considered a face. This threshold was set to 7 as the Haar Cascades are incredibly sensitive to the parameter choices and after trial and error, 7 was the value that minimized the number of false-positive detections. The final parameter is _minSize_ which indicates the window’s minimum size. To specify, _minSize_ is a tuple of width and height in pixel units that serves as the minimum dimensional threshold to which a bounding box can be considered valid.

### FCHD

The goal of this section is to outline the process in which we can fine-tune an FCHD model. As noted by Vora et al., deep architectures are powerful as they are capable of adapting to “generalized features” that have high applicability. The [FCHD Model](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector) employs a VGG16 model as its base model. The VGG16 is considered a powerful architecture as it has been proven to be able to classify 1000 images of 1000 different categories with 92.7\% accuracy. Due to its public availability and ease to apply to transfer learning, it is one of the most popular deep learning algorithms for image classification.

The architecture of Vora et al.’s FCHD head detection model adds three new convolutional layers to the VGG16 model to allow for further hierarchical decomposition of the input. For the purposes of our experiment, given that we are employing the same FCHD model, we set identical hyperparameters to Vora et al. The pre-trained VGG16 model is trained using the ImageNet dataset, and the new layers are initialized with random weights sampled from a standard normal distribution with $\mu=0$ and $\sigma=0.01$. Other hyperparameters include a weight decay of $0.0005$, a learning rate for the training set to $0.001$, and an epoch count of $15$. Note that the whole training uses a PyTorch framework.

Now, to discuss the reasons we chose to employ an FCHD model, Vora et al.’s model specializes in head detection for crowded scenes. Given that the Gaze Data contains scenes where multiple characters from a movie scene are conglomerated in one area, sophisticated deep learning frameworks that can detect faces under challenging conditions (such as high occlusion) was necessary.

### DeepFace

The methodology for implementing DeepFace is incredibly simple. Given that the face recognition pipeline is offered as a PyPI package, the only requirements include installing the library itself and its prerequisites. The advantages of this library are obvious. As aforementioned, DeepFace offers all procedures necessary for face recognition including: detect, align, normalize, represent, and verify. For the purposes of this research, we will use the Face Recognition algorithm they offer to analyze the cosine similarity between images and to configure a localization function. DeepFace serves as a good evaluation metric for understanding the performances of the previous two architectures as well. In addition to the conventional statistical metrics: accuracy, precision, and recall, employing DeepFace will allow us to identify a measure of result relevancy and understand the number of truly relevant features that the model returns.


## Results/ Discussions / Conclusion
Please reference the [research paper](https://github.com/staniya/cs153_automated_labels/blob/main/Taniya_Shinn_Report.pdf)