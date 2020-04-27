import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10

class Settings:
    
    # constants
    NB_EPOCHS = 6
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NB_FILTERS = 64
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    NUM_THREADS = None
    LABEL_SMOOTHING = 0.1

    # basic settings
    """
    "attack_type"
    fgsm -> fast gradient sign method
    pgd -> projected gradient descent
    bim -> basic iterative method
    cwl2 -> Carlini & Wagner L2
    jsma -> jsma method

    "dataset"
    mnist -> mnist dataset
    fmnist -> fashion mnist dataset
    cifar10 -> cifar-10 dataset

    "attack_model"
    basic_model -> a cnn model for mnist
    all_cnn -> a cnn model for cifar10
    cifar10_model -> model for cifar10
    mnist_model -> model for mnist
    """
    attack_type = ['fgsm', 'pgd'] 
    eval_attack_type = ['fgsm', 'pgd', 'spsa']
    dataset = 'mnist' 
    attack_model = 'basic_model'
    exp_name = 'pgd_as_spsa'
    
    # advanced settings 
    REINFORE_ENS = [] 
    PRED_RANDOM = False
    RANDOM_K = 3
    RANDOM_STDDEV = 0.1
    IS_ONLINE = False
    LINEAR_DETECTOR = False
    EVAL_DETECTOR = True
    BLACKBOX_SAMPLES_METHOD = 'pgd'
    SEPRATED_DETECTOR_LOSS = False

    # static varibales 
    def_list_addon = 1 if SEPRATED_DETECTOR_LOSS else 2
    blackbox_samples = 'blackbox_attack_adv_samples_by_' + BLACKBOX_SAMPLES_METHOD + '.npy'
    exp_path = './exps/' + dataset + '_' + exp_name + '/' + attack_model
    saved_data_path = exp_path + '/train_data'
    Path(saved_data_path).mkdir(parents=True, exist_ok=True)
    fp = open(exp_path + '/' + str(IS_ONLINE) + '.txt', 'w')

    rng = np.random.RandomState([2017, 10, 30])
    
    if NUM_THREADS:
        config_args = dict(intra_op_parallelism_threads=1,
                                allow_soft_placement=True, log_device_placement=False)
    else:
        config_args = dict(allow_soft_placement=True, log_device_placement=False)

    train_params = {
        'nb_epochs': NB_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }

    eval_params = {'batch_size': 128}

        # Set paramters for different dataset
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                    train_end=TRAIN_SIZE,
                                                    test_start=0,
                                                    test_end=TEST_SIZE)
        assert Y_train.shape[1] == 10

        nb_classes = 10
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        input_shape = [28, 28, 1]

    elif dataset == 'cifar10':
        data = CIFAR10(train_start=0, train_end=TRAIN_SIZE,
                    test_start=0, test_end=TEST_SIZE)
        X_train, Y_train = data.get_set('train')
        X_test, Y_test = data.get_set('test')
        assert Y_test.shape[1] == 10.

        nb_classes = Y_test.shape[1]
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        input_shape = [32, 32, 3]
