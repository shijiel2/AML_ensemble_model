"""
Use unweighted to attack weighted
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import numpy as np
import logging
import math
import itertools
from tqdm import tqdm


import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.attacks import *
from cleverhans.train import *
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10
from cleverhans.model import CallableModelWrapper
#from utils_fmnist import data_fmnist
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
from models.cifar10_model import make_wresnet
from models.mnist_model import MadryMNIST
from models.simple_model import SimpleLinear
from models.simple_Cifar10 import SimpleCifar10
from sklearn.model_selection import train_test_split
from generate import *

FLAGS = flags.FLAGS


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
TRAIN_SIZE = 60000
TEST_SIZE = 10000
IS_ONLINE = False
CHECK_DET = False
NOISE = True
NOISE_NB = 5
NOISE_STD = 0.1
ENS_IN_WEI = True
ENS_PL = 'probs'


def defence_frame(train_start=0, train_end=TRAIN_SIZE, test_start=0,
                  test_end=TEST_SIZE, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  clean_train=CLEAN_TRAIN,
                  testing=False,
                  backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                  nb_filters=NB_FILTERS, num_threads=None,
                  label_smoothing=0.1):

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1,
                           allow_soft_placement=True, log_device_placement=True)
    else:
        config_args = dict(allow_soft_placement=True,
                           log_device_placement=True)
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Set parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    train_params_det = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': 0.01
    }
    eval_params = {'batch_size': FLAGS.batch_size}

    def_model_list = []


    '''
    -------------- initial dataset --------------
    '''
    if FLAGS.dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)
        assert Y_train.shape[1] == 10
        nb_classes = 10
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        input_shape = [28, 28, 1]
        
    elif FLAGS.dataset == 'cifar10':

        data = CIFAR10(train_start=train_start, train_end=train_end,
                       test_start=test_start, test_end=test_end)
        dataset_size = data.x_train.shape[0]
        X_train, Y_train = data.get_set('train')
        X_test, Y_test = data.get_set('test')

        # Use Image Parameters
        img_rows, img_cols, nchannels = X_test.shape[1:4]
        nb_classes = Y_test.shape[1]
        assert Y_test.shape[1] == 10.

        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        input_shape = [32, 32, 3]

    '''
    ----------------- create model0, attack model and train
    '''
    att_list = []
    model_list = []
    rng = np.random.RandomState([2017, 10, 30])

    # define and train clean model to be defenced on
    model_c = get_model(FLAGS.dataset, FLAGS.origin_model,
                      'model', nb_classes, nb_filters, input_shape)
    loss_c = CrossEntropy(model_c, smoothing=label_smoothing)

    train(sess, loss_c, X_train, Y_train,
              args=train_params, rng=rng, var_list=model_c.get_params())

    # for the 1...M attack methods, create adv samples and train defence models
    for i, model_name in enumerate(FLAGS.attack_model):
        model_i = get_model(FLAGS.dataset, model_name,
                      'model_'+str(i), nb_classes, nb_filters, input_shape)
        loss_i = CrossEntropy(model_i, smoothing=label_smoothing)
        train(sess, loss_i, X_train, Y_train,
          args=train_params, rng=rng, var_list=model_i.get_params())
        for attack_name in FLAGS.attack_type:
            print(model_name, attack_name)
            attack_params = get_para(FLAGS.dataset, attack_name)
            attack_method = get_attack(attack_name, model_i, sess)

            def attack(x):
                return attack_method.generate(x, **attack_params)
                
            att_list.append(attack)

    att = att_list[np.random.randint(len(att_list))]

    model_ens = get_model(FLAGS.dataset, FLAGS.origin_model,
                      'model_ens', nb_classes, nb_filters, input_shape)
    loss_ens = CrossEntropy(model_ens, smoothing=label_smoothing)

    loss_ens_2 = CrossEntropy(model_ens, smoothing=label_smoothing, attack = att)

    train_ens(sess, loss_ens, loss_ens_2, X_train, Y_train,
              args=train_params, rng=rng, var_list=model_ens.get_params())



    # Evaluate the accuracy of model on clean examples
    print('----------- clean examples ----------')
    do_eval(sess, x, y, do_probs(x, model_c), 
            X_test, Y_test, "origin model on clean data", eval_params)
    do_eval(sess, x, y, do_probs(x, model_ens), 
            X_test, Y_test, "origin model on clean data", eval_params)  

    # Evaluate the accuracy of model on adv examples
    print('----------- test on ens examples ----------')
    for i, attack_name in enumerate(FLAGS.attack_type):
        origin_attack = get_attack(attack_name, model_c, sess)
        origin_adv_x = origin_attack.generate(x, **attack_params)

        ens_attack = get_attack(attack_name, model_ens, sess)
        ens_adv_x = ens_attack.generate(x, **attack_params)        
        # check the weights from the detector for model0
        do_eval(sess, x, y, do_probs(origin_adv_x, model_ens), 
                X_test, Y_test, attack_name + "-> origin model, test on ens model", eval_params)
        do_eval(sess, x, y, do_probs(ens_adv_x, model_ens), 
                X_test, Y_test, attack_name + "-> origin model, test on ens model", eval_params)
'''
------------ help function ---------------
'''

def train_attack_detector(sess, x, det_model,adv_x, X_train, Y_train, train_params, rng, label_smoothing=0.1):
    
    nb_imgs = Y_train.shape[0]    
    nb_model = len(adv_x) 
    loss_det = CrossEntropy(det_model, smoothing=label_smoothing)

    X_class = np.zeros((nb_imgs*nb_model,) + X_train.shape[1:], dtype = X_train.dtype)
    Y_class = np.zeros((nb_imgs*nb_model,nb_model), dtype = Y_train.dtype)
    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_train)) / train_params['batch_size']))
        assert nb_batches * train_params['batch_size'] >= len(X_train)

        for i, att in enumerate(adv_x.keys()): 
            print(i,att)
            for batch in tqdm(range(nb_batches)):
                start = batch * train_params['batch_size']
                end = min(len(X_train), start + train_params['batch_size'])
                feed_dict = {x: X_train[start:end]}
                X_class[i*nb_imgs+start:i*nb_imgs+end]  = adv_x[att].eval(feed_dict = feed_dict)

            Y_class[i*nb_imgs:(i+1)*nb_imgs,i] = 1.

        if CHECK_DET == True:
            eval_params = {'batch_size': FLAGS.batch_size}

            X_class_train, X_class_test, Y_class_train, Y_class_test = train_test_split(
                X_class, Y_class, test_size=0.20, random_state=42)
            train(sess, loss_det, X_class_train, Y_class_train,
                  args=train_params, rng=rng, var_list=det_model.get_params())
            do_eval(sess, x, tf.placeholder(tf.float32, shape=(None,nb_model)),
                    do_probs(x, det_model), X_class_test, Y_class_test, 'Detector accuracy', eval_params)
        else:
            train(sess, loss_det, X_class, Y_class,
                  args=train_params, rng=rng, var_list = det_model.get_params())


def do_eval(sess, x, y, pred, X_test, Y_test, message, eval_params):
    print(message)
    acc = model_eval(
        sess, x, y, pred, X_test, Y_test, args=eval_params)
    print('Accuracy: %0.4f\n' % acc)
    return acc


"""
Return the mean prob of multiple models
"""
def do_probs(x, model, detector=None, def_model_list=None):
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    if detector is None:
        preds = tf.stack([m.get_probs(x) for m in models], 1)
        return tf.reduce_mean(preds, 1)
    else:
        preds = tf.stack([m.get_probs(x) for m in models], 1)
        if NOISE == True:
            preds = tf.math.multiply(preds, do_det_noise(detector,x)[:, :, None])
        else:
            preds = tf.math.multiply(preds, detector.get_probs(x)[:, :, None])
        return tf.reduce_sum(preds, 1)

"""
Return the mean logits of multiple models
"""
def do_logits(x, model, detector=None, def_model_list=None):
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    if detector is None:
        preds = tf.stack([m.get_logits(x) for m in models], 1)
        return tf.reduce_mean(preds, 1)
    else:
        preds = tf.stack([m.get_logits(x) for m in models], 1)
        if NOISE == True:
            preds = tf.math.multiply(preds, do_det_noise(detector,x)[:, :, None])
        else: 
            preds = tf.math.multiply(preds, detector.get_probs(x)[:, :, None])
        return tf.reduce_sum(preds, 1)

def do_det_noise(detector,x):
    nb_noise = NOISE_NB
    noise_x = []
    for i in range(nb_noise):
        noise_x.append(x+tf.random_normal(shape = tf.shape(x), 
                                mean=0.0, stddev=NOISE_STD, dtype=tf.float32))
    det = tf.stack([detector.get_probs(noi) for noi in noise_x], 0)
    return tf.reduce_mean(det, 0)


def main(argv):
    #  from cleverhans_tutorials import check_installation
    #  check_installation(__file__)

    defence_frame(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                  learning_rate=FLAGS.learning_rate,
                  clean_train=FLAGS.clean_train,
                  backprop_through_attack=FLAGS.backprop_through_attack,
                  nb_filters=FLAGS.nb_filters)
    

if __name__ == '__main__':

    flags.DEFINE_float(
        'label_smooth', 0.1, ("Amount to subtract from correct label "
                              "and distribute among other labels"))

    flags.DEFINE_list('attack_type', ['fgsm','pgd'], ("Attack type: 'fgsm'->'fast gradient sign method', "
                                                       "'pgd'->'projected gradient descent', "
                                                       "'bim'->'basic iterative method',"
                                                       "'cwl2'->'Carlini & Wagner L2',"
                                                       "'jsma'->'jsma method'"))
    flags.DEFINE_string('dataset', 'mnist',
                        ("dataset: 'mnist'->'mnist dataset', "
                         "'fmnist'->'fashion mnist dataset', "
                         "'cifar10'->'cifar-10 dataset'"))
    flags.DEFINE_string('origin_model', 'mnist_model',
                        ("defence_model: 'basic_model'->'a cnn model for mnist', "
                         "'all_cnn'->'a cnn model for cifar10', "
                         "'cifar10_model'->'model for cifar10', "
                         "'mnist_model'->'model for mnist'"))
    flags.DEFINE_list('attack_model', ['basic_model', 'mnist_model'],
                        ("defence_model: 'basic_model'->'a cnn model for mnist', "
                         "'all_cnn'->'a cnn model for cifar10', "
                         "'cifar10_model'->'model for cifar10', "
                         "'mnist_model'->'model for mnist'"))
    flags.DEFINE_integer('nb_filters', NB_FILTERS,
                         'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                         'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    app.run(main)
