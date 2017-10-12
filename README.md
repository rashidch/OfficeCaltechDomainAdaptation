# OfficeCaltechDomainAdaptation

## What is this ?
This dataset is part of the Computer Vision problematic consisting in making machines learn to detect the presence of an object in an image. Here, we want to learn a classification model that takes as input an image and return the category of the object it contains.

The Office Caltech dataset contains four different domains: amazon, caltech10, dslr and webcam. These domains contain respectively 958, 1123, 295 and 157 images. Each image contains only one object among a list of 10 objects: backpack, bike, calculator, headphones, keyboard, laptop, monitor, mouse, mug and projector.

With this benchmark dataset in Domain Adaptation, we repeatedly take one of the four domains as Source domain S and one of the three remaining as target T. The aim is then to learn to classify images with the data from S to correctly classify the images in T.

## What is available in this repository ?
In addition to the images, we also give features that were extracted from the images to describe them. We give different sets of features that describe all the images in the corresponding folder.

We propose some code in python3 to show how to evaluate the benchmark. What is usually evaluated with this benchmark are Domain Adaptation algorithms. We provide code for a few of them.

## Dependencies
Python3 and some python3 libraries:
 - numpy
 - scipy
 - sklearn

## Example of execution

Program launched by executing the main.py script with python:
```
python3 main.py
```

For each adaptation problem among the 12 possible, each adaptation algorithm chosen at the beginning of the file is applied. Then are reported the mean accuracy and standard deviation. Results (using the default surf [1] features):
```
Feature used:  surf
Number of iterations:  10
Adaptation algorithms used:   NA  SA  OT
A->C ..........   3.42s
     22.6  2.0 NA
     33.9  1.6 SA
     29.7  0.9 OT
A->D ..........   1.03s
     22.2  2.9 NA
     31.7  3.8 SA
     40.2  2.9 OT
A->W ..........   1.35s
     24.9  2.8 NA
     31.3  2.5 SA
     35.3  2.1 OT
C->A ..........   3.09s
     21.7  2.5 NA
     36.3  2.3 SA
     36.6  3.2 OT
C->D ..........   0.92s
     21.9  4.1 NA
     35.5  4.7 SA
     42.0  3.8 OT
C->W ..........   1.24s
     19.4  2.1 NA
     30.1  3.3 SA
     38.4  4.9 OT
D->A ..........   2.11s
     27.3  2.9 NA
     32.4  1.7 SA
     29.2  1.9 OT
D->C ..........   2.51s
     25.3  1.2 NA
     30.1  1.3 SA
     29.9  1.2 OT
D->W ..........   0.83s
     52.0  2.5 NA
     78.6  2.5 SA
     68.7  2.1 OT
W->A ..........   2.96s
     22.8  1.6 NA
     32.1  1.7 SA
     37.3  0.7 OT
W->C ..........   3.50s
     18.9  1.1 NA
     29.2  1.0 SA
     33.9  1.1 OT
W->D ..........   0.88s
     52.5  3.5 NA
     83.1  1.9 SA
     72.0  1.9 OT

Mean results:
     27.6  2.4 NA
     40.4  2.4 SA
     41.1  2.2 OT
```

By modifying the feature used in the script with CaffeNet [2] features:
```
Feature used:  CaffeNet4096
Number of iterations:  10
Adaptation algorithms used:   NA  SA  OT
A->C ..........  14.09s
     71.8  2.8 NA
     78.6  1.8 SA
     82.7  0.6 OT
A->D ..........   5.05s
     78.5  4.0 NA
     82.5  2.8 SA
     93.4  1.3 OT
A->W ..........   6.19s
     68.5  3.1 NA
     78.8  3.0 SA
     92.4  0.7 OT
C->A ..........  12.72s
     81.6  2.1 NA
     85.4  1.4 SA
     88.2  1.5 OT
C->D ..........   5.06s
     78.0  3.1 NA
     81.5  1.9 SA
     93.0  1.2 OT
C->W ..........   6.24s
     70.5  4.1 NA
     78.0  3.6 SA
     89.0  1.8 OT
D->A ..........  11.31s
     70.5  2.0 NA
     83.3  1.3 SA
     86.9  1.0 OT
D->C ..........  12.60s
     66.2  1.3 NA
     75.7  1.0 SA
     81.4  1.4 OT
D->W ..........   5.33s
     91.9  1.1 NA
     96.5  1.1 SA
     96.4  0.5 OT
W->A ..........  12.89s
     69.2  3.0 NA
     82.4  0.6 SA
     86.8  1.7 OT
W->C ..........  14.21s
     60.4  1.0 NA
     73.7  0.6 SA
     76.9  1.8 OT
W->D ..........   5.01s
     95.9  1.5 NA
     99.4  0.5 SA
     97.3  0.8 OT

Mean results:
     75.3  2.4 NA
     83.0  1.6 SA
     88.7  1.2 OT
```

and with GoogleNet [3] features:
```
Feature used:  GoogleNet1024
Number of iterations:  10
Adaptation algorithms used:   NA  SA  OT
A->C ..........   4.29s
     84.7  1.4 NA
     85.5  1.3 SA
     89.6  0.6 OT
A->D ..........   1.15s
     88.4  1.2 NA
     88.5  2.3 SA
     93.7  0.7 OT
A->W ..........   1.60s
     82.4  2.5 NA
     84.2  2.5 SA
     95.7  0.9 OT
C->A ..........   3.90s
     91.0  0.8 NA
     91.3  0.6 SA
     93.6  0.5 OT
C->D ..........   1.18s
     87.7  2.4 NA
     89.3  2.6 SA
     93.9  1.1 OT
C->W ..........   1.56s
     83.9  3.9 NA
     88.1  2.6 SA
     96.8  0.5 OT
D->A ..........   2.75s
     83.4  1.4 NA
     88.5  1.3 SA
     91.2  0.8 OT
D->C ..........   3.25s
     76.6  1.7 NA
     83.2  1.0 SA
     89.6  0.5 OT
D->W ..........   1.07s
     97.0  0.7 NA
     98.0  1.0 SA
     98.0  0.6 OT
W->A ..........   3.85s
     86.9  1.2 NA
     90.1  0.3 SA
     92.8  0.2 OT
W->C ..........   4.32s
     80.4  1.3 NA
     84.5  1.1 SA
     90.1  0.8 OT
W->D ..........   1.16s
     99.3  0.4 NA
     99.4  0.2 SA
     97.4  1.0 OT

Mean results:
     86.8  1.6 NA
     89.2  1.4 SA
     93.5  0.7 OT
```
[1] Gong, B., Grauman, K., & Sha, F. (2014). Learning kernels for unsupervised domain adaptation with applications to visual object recognition. International Journal of Computer Vision, 109(1-2), 3-27.

[2] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., ... & Darrell, T. (2014, November). Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 675-678). ACM.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
