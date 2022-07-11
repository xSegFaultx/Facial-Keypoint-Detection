# Facial Keypoint Detection
[![Udacity Computer Vision Nanodegree](https://img.shields.io/badge/Udacity-Computer%20Vision%20ND-deepskyblue?style=flat&logo=udacity)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)
[![Pytorch](https://img.shields.io/badge/%20-Pytorch-grey?style=flat&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/%20-OpenCV-grey?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAtCAYAAADoSujCAAAFTElEQVR42mKgFPzTcQEUXw8wlnNhGMc7tj+tbdu2bdu2bdu2bdu2bRuD8/076U1umj3tDu7dTX5ZvtvnPT2qIlIiVWF3fi6ASbiAjwhDCJ5jNzoiMTUOYakKK3/8hxpCDUOoTFiMNxAGQnAd3fAXlNA0Rf5MeEF4GnAmRAM8hIiAUOxAGqj/zx8J78TD2+IjRCSdQwaob9J+04YHqiriLUQU7Ucs2KcBLXw8nIeIJiMZGEd+tsPopwhvoK/ZYsVDHMVZvIMw8AwZYZfR/w8XTMJ0RSLWijc/ByAnluGnQd0QdXAYJJs3UBzfJCHeoOKHPCUUoQUJTl/EUueDGRASx+AHm2S3DtINQmISu4kj4WXNJ8QNSe1zpLRRA9qJmzw8xBRJgO8oDun6CUlbRD305kjqv6IAbLf3E0BtYIEkwDtkkgUIThdeqxosqf+JUjZ+A4XUANMNRrAQ5DtY4vD6aZL6byhkswa00Vf1hpAYJmLkVwzWQEyDHewV0sDmu1BZ/JCEeIIC+u1Qu++48vNQhElqzyDQHg3EwXUIiWtIDcvUsxx+7fDNoG4cU0xr3IYn8dtCJdUwI00uaHGsG9Aar43vBudHLrtcr7UwKXBLspXWg34KqevCi5/XShqYwb930daO3e5DjfFVF2QjfPTzmFG1NJ4bL3/xxpLY9UqtjagbPw+0WtCvkV8WRB1d7cY5BkJzB/l1b8x+TcAD3fEOEwnhRFCzmsS4ivPIJ5LbKPyom80sXBETqZEZKfEfXC63rKCIZIWc1SCID/XO9Dvf0HmQ2CYjr4V2QAK0xgbcwCu8x0tcx1o0G3WrWdx+r1o5DHnUwo7v33jE/dABNxEKYWIdvGH6jMAxwiIA5eAH0xp4oxL+hQJp+PhYhWAIMwhGSyj43fBxsRKfUQ8KzGqq4jO2IZl1jXX4uNgFEQE74R+B8LGwFUJzFnFMGvgXRyE0h5EYisoS3guLIhj+PYpHYPQ9MAdCZwAUyJruilBdzWr4WTfQED8gImAmXCIw+jXxDULnMTJDgb4uFe5C6ASjpaWBGDhhEvY19mM1juAc0kRg9ANwEEJiCdxgXeOMGdIabfqpARrguyR4GDYjBzzhBF/EgWMERr8kvkJIrICPrgF3zEKYpOYHmqsNrIGQ2IZ/zUfatIGREBJ7EAMK9HWBWGdQu0kNdlcS/gMKmYc3bcANWyUBvqK0ySLOjXeS+gdquK+SBk4jAGYHnyd8f8FLkClgjPDnQWckAa7hP5Nt1A8nZNNIMTht98DdpAFPzMYpnLByCsun3K/p5zVCBPGgS5IAJ+Fr0oA7dkvqw9QQnyUNXMDfJg3EwW3ZG5x4r66fxzDhYzCCdxHPpIG/cUFS/0UNcUMS4AvKQoFs+tTDT0n9BhpwcR8avh2ulQQIRkOTNVDBYAe7oYZYDCFxFImgQB8+HS4Z1PYdebO5EjQmTA3RG0LiKjJJdqHkOGlQu1ANUg1fICSOoQL+hQ9iohYuGtS8Qy5YguSU7SSaa2iM+AhAHNTEWYOaz6isNhCEfRAGvuIqjuMGvkMYWAsPWN+D1kAY+In7OI+7+AFhYDcCLNOhknQxR9wr5IeiNWBRAK8gosFHlINi/dk4AWFRDP8TPeEI/YJ0RC/8jGL4UIyCi/6b4C8sj0ITwZgILxh9WU1CcBTCz0cAfrkt/o1J8kUt9Q79TD4trZvoi7cRDP8JoyzhIT2c3FENh/HNJPhnbEcxOEfgu9gZhbEJH02Cf8ZulIWr5NCTvo3aWIV7VjvPN9zCIlSEHxRE5pbqg1KYjvN4gy94i0uYiwrwl436/3QR5G/eVcYGAAAAAElFTkSuQmCC)](https://opencv.org/) \
In this project, I designed a robust end-to-end facial keypoint detection system utilizing PyTorch and OpenCV. 
The system takes an image (arbitrary size) as input and is able to locate and detect facial keypoints for every single face presented in the given image. 
One example output of the system is shown in Fig 1.0.
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig1.0.png" alt="input and output of the pipeline">
<figcaption align = "center"><b>Fig.1.0 - Input and output of the system. Left: Input image; Right: Output Image </b></figcaption>
</figure>

# Facial Keypoint Detection Pipline
The goal of this project is to detect keypoints of all faces in an arbitrary-size input image. 
Usually, keypoints detection is done by using convolutional neural networks. 
However, for this project, a single CNN may not be suitable since:
1. CNN usually takes a fixed-size input
2. CNN usually has a fixed-size output which means only a fixed number of keypoints can be predicted.

To solve this problem, a facial detection and localization module is utilized. 
This module will detect all the faces in the image and crop each face from the origin image. 
The cropped images will have fixed size and will be passed to the CNN one by one. 
Since the number of keypoints in each face is fixed, the CNN can have a fixed number of outputs.

In summary, the facial keypoint detection pipeline consists of two parts:
1. A human face detection and localization module that can detect and extract human faces that are presented in a given image.
2. A keypoint detection network that can predict the locations of facial keypoints given an image of a human face.
## Human Face Detection and Localization
The "Human face detection and localization module" is responsible for detecting and extracting human faces in a given image. 
To achieve this, I used the pre-trained Haar Cascade classifiers from the OpenCV library. 
The Haar Cascade classifier will detect human faces in a given image and return the location of each human face as shown in Fig 2.0. 
After knowing the locations of the faces, each face can be cropped out from the input image and passed to the "Keypoint Detection Network" as shown in Fig 2.1.
Each area within the bounding box will be cropped out of the origin image and passed to the "Keypoint Detection Network" for keypoint detection.
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig2.0.png" alt="Haar Cascade classifier output">
<figcaption align = "center"><b>Fig.2.0 - Faces that are detected by the Haar Cascade classifier </b></figcaption>
</figure>

<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig2.1.png" alt="Faces extracted from the input image">
<figcaption align = "center"><b>Fig.2.1 - Faces extracted from the input image </b></figcaption>
</figure>

## Keypoint Detection Network
The keypoint detection network is a convolutional neural network that takes a fixed-size image of a human face and predicts the locations of facial keypoints. 
The initial model design follows the model structure described in ["Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet"](https://arxiv.org/pdf/1710.00977.pdf) by Agarwal, N., Krohn-Grimberghe, A., and Vyas, R. In this paper, Agarwal et al., developed a modified version of "LeNet" to detect 15 facial keypoints given an image of a human face. 
While the model performed quite well on detecting 15 keypoints, it performed poorly on the [Youtube Face dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/) that I used for this project which requires the model to detect 68 facial keypoints. 
Since the ["LeNet"](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) has a rather "shallow" structure, it is not capable of extracting high-level features, such as the entire eyebrows or the entire nose, which are crucial for the model to detect all 68 keypoints. 
To solve this problem, I designed a modified version of the [VGG16](https://arxiv.org/pdf/1409.1556.pdf) model which is deeper and has more parameters than the proposed model from the paper mentioned above. 
The detailed model structure is shown in Fig 3.0.
With the additional help from the small receptive field and batch normalization, the model performed quite well on the YouTube Face dataset. 
A few example outputs of the model are shown in Fig 3.1.
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig3.0.png" alt="CNN structure">
<figcaption align = "center"><b>Fig.3.0 - CNN structure </b></figcaption>
</figure>
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig3.1.png" alt="Keypoints prediction">
<figcaption align = "center"><b>Fig.3.1 - Facial keypoints predicted by the trained model </b></figcaption>
</figure>

# Result
The Keypoint Detection Network takes around 50 epochs to converge. 
After the convergence of the model and some fine-tuning on the Facial detection and localization module, I can finally put together the final Facial Keypoint Detection pipeline. 
The output of the pipeline is shown in Fig 4.0 below.
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig4.0a.png" alt="Input image">
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig4.0b.png" alt="Keypoints prediction">
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig4.0c.png" alt="Output image">
<figcaption align = "center"><b>Fig.4.0 - Top: Input image Middle: Facial Keypoints predicted by the trained model Bottom: Output image</b></figcaption>
</figure>

# Applications
Nowadays, many apps use Facial Keypoints Detection systems in many different ways. 
For example, Apple's FaceID uses a more advanced version of the Facial Keypoints Detection system, filters and special effects in TikTok, SnapChat, and Instagram utilize Facial Keypoints Detection, and some photo editing software could even use Facial Keypoints to automatically apply makeups. 
To demonstrate the usefulness and effectiveness of my pipeline, I've created a filter that will apply the "sunglasses" special effect on users' faces base on the detected facial keypoints. 
I've used Dr. Yann LeCun's profile picture as the input image and the output image is shown in Fig 5.0 below.
<figure>
<img src="https://github.com/xSegFaultx/Facial-Keypoint-Detection/raw/master/images/fig5.0.png" alt="Sunglasses effect">
<figcaption align = "center"><b>Fig.5.0 - Left: Input image Middle: Keypoints prediction Right: Output image</b></figcaption>
</figure>


# WARNING
This is a project from Udacity's ["Computer Vision Nanodegree"](https://www.udacity.com/course/computer-vision-nanodegree--nd891). I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.
