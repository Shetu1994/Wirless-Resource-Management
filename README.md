# Learning to optimize: Training deep neural networks for wireless resource management.
Python code to reproduce our works on DNN research for SPAWC 2017. 

Demo.py contains the whole process from data generation, training, testing to plotting for 10 users' IC case, even though such process done on a small dataset of 25000 samples, 94% accuracy can still be easily attained in less than 100 iterations.

In test.py, we do the testing stage for Table I: Gaussian IC case in the paper, the testing are based on the pre-trained models. To train models from scratch, please follow the instructions in the paper and read the demo.py for reference.

All codes have been tested successfully on Python 3.6.0 with TensorFlow 1.0.0 and Numpy 1.12.0 support.

References: 
[1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu, and Nikos D. Sidiropoulos. "Learning to optimize: Training deep neural networks for wireless resource management." in proceedings of the 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC),  IEEE, 2017.

version 1.0 -- February 2017. 

Written by Haoran Sun and Xiangyi Chen (sun00111@umn.edu, chen5719@umn.edu)
