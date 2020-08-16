# Overfitting and Regularization for wireless resource management.
Python code to reproduce our works on DNN research for SPAWC 2017. 

Demo.py contains the whole process from data generation, training, testing to plotting for 10 users' IC case, even though such process done on a small dataset of 25000 samples, 94% accuracy can still be easily attained in less than 100 iterations.

In test.py, we do the testing stage for Table I: Gaussian IC case in the paper, the testing are based on the pre-trained models. To train models from scratch, please follow the instructions in the paper and read the demo.py for reference.

All codes have been tested successfully on Python 3.6.0 with TensorFlow 1.0.0 and Numpy 1.12.0 support.


