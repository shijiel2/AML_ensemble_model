import tensorflow as tf
import sys
import numpy as np
from settings import Settings
from sklearn import svm

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)