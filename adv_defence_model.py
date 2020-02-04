"""Runs CleverHans attacks on the Madry Lab MNIST challenge model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import numpy as np
import logging

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.attacks import *
from cleverhans.train import train
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10
from cleverhans.model import CallableModelWrapper
#from utils_fmnist import data_fmnist
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.loss import CrossEntropy
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
from models.cifar10_model import make_wresnet
from models.mnist_model import MadryMNIST
from utils import do_eval, do_probs, do_logits, do_transform

FLAGS = flags.FLAGS


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
TRAIN_SIZE = 60000
TEST_SIZE = 10000
ENSEMBLE_TYPE = 'probs'
IS_ONLINE = True


def defence_frame(train_start=0, train_end=TRAIN_SIZE, test_start=0,
                  test_end=TEST_SIZE, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  clean_train=CLEAN_TRAIN,
                  testing=False,
                  backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                  nb_filters=NB_FILTERS, num_threads=None,
                  label_smoothing=0.1):

    """
    **************************** Settings & Parameters ****************************
    """              

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
    eval_params = {'batch_size': FLAGS.batch_size}
    # list of model_1...model_n
    def_model_list = []
    attack_dict = {}

    # Set paramters for different dataset
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

    """
    **************************** Logic Operations ****************************
    """

    # Define and train model (aka model_0)
    model = get_model(FLAGS.dataset, FLAGS.attack_model, 'model', nb_classes, nb_filters,
                      input_shape)
    rng = np.random.RandomState([2017, 10, 30])
    loss = CrossEntropy(model, smoothing=label_smoothing)
    train(sess, loss, X_train, Y_train,
          args=train_params, rng=rng, var_list=model.get_params())

    # For the 1...N attack methods, create adv samples and train defence models model_1...model_n
    for i, attack_name in enumerate(FLAGS.attack_type):
        attack_params = get_para(FLAGS.dataset, attack_name)
        model_i = get_model(FLAGS.dataset, FLAGS.attack_model,
                            'model_'+str(i), nb_classes, nb_filters, input_shape)
        if IS_ONLINE:
            attack_method = get_attack(attack_name, model_i, sess)
        else:
            attack_method = get_attack(attack_name, model, sess)

        def attack(x):
            return attack_method.generate(x, **attack_params)
        attack_dict[attack_name] = attack

        loss_i = CrossEntropy(
            model_i, smoothing=label_smoothing, attack=attack, adv_coeff=1.)
        train(sess, loss_i, X_train, Y_train,
              args=train_params, rng=rng, var_list=model_i.get_params())

        def_model_list.append(model_i)


    # Make and train detector that classfies whcih source (clean or adv_1 or adv_2...) 
    # the input comes from in the form of probs. The probs will be used as weights in 
    # ensemble model.
    
    # prepare data
    X_merged, Y_merged = get_merged_train_data(sess, x, attack_dict, X_train, eval_params)
    # train the detector
    detector = ModelBasicCNN('detector', (len(FLAGS.attack_type) + 1), nb_filters)
    loss_d = CrossEntropy(detector, smoothing=label_smoothing)
    train(sess, loss_d, X_merged, Y_merged,
          args=train_params, rng=rng, var_list=detector.get_params())
    pred = detector.get_probs(attack_dict['fgsm'](x))
    with sess.as_default():
        print(pred.eval(feed_dict={x:X_test[:3]}))
    return

    # Make Ensemble model
    def ensemble_model_logits(x):
        return do_logits(x, model, def_model_list=def_model_list)
    def ensemble_model_probs(x):
        return tf.math.log(do_probs(x, model, def_model_list=def_model_list))

    if ENSEMBLE_TYPE == 'logits':
        ensemble_model = CallableModelWrapper(ensemble_model_logits, 'logits')
    elif ENSEMBLE_TYPE == 'probs':
        ensemble_model = CallableModelWrapper(ensemble_model_probs, 'logits')

    # Evaluate the accuracy of model on clean examples
    do_eval(sess, x, y, do_probs(x, model), 
            X_test, Y_test, "origin model on clean data", eval_params)
    do_eval(sess, x, y, do_probs(x, ensemble_model), 
            X_test, Y_test, "ensemble model on clean data", eval_params)

    # Evaluate the accuracy of model on adv examples
    for attack_name in FLAGS.attack_type:
        attack_params = get_para(FLAGS.dataset, attack_name)

        # generate attack to origin model
        origin_attack = get_attack(attack_name, model, sess)
        origin_adv_x = origin_attack.generate(x, **attack_params)
        
        do_eval(sess, x, y, do_probs(origin_adv_x, model), 
                X_test, Y_test, attack_name + "-> origin model, test on origin model", eval_params)
        do_eval(sess, x, y, do_probs(origin_adv_x, ensemble_model), 
                X_test, Y_test, attack_name + "-> origin model, test on ensemble model", eval_params)
        
        # generate attack to ensemble model
        ensemble_attack = get_attack(attack_name, ensemble_model, sess)
        ensemble_adv_x = ensemble_attack.generate(x, **attack_params)
        
        do_eval(sess, x, y, do_probs(ensemble_adv_x, model), 
            X_test, Y_test, attack_name + "-> ensemble model, test on origin model", eval_params)
        do_eval(sess, x, y, do_probs(ensemble_adv_x, ensemble_model), 
            X_test, Y_test, attack_name + "-> ensemble model, test on ensemble model", eval_params)
        
"""
**************************** Helper Functions ****************************
"""
def get_merged_train_data(sess, x, attack_dict, X_train, eval_params):
    """
    Construct the merged data for trainning the classifier
    :param sess: current session
    :param x: placeholder for trainning data
    :param attack_dict: dictionary {attack_name: attack_generator}
    :param X_train: ndarray 
    :return: X_merged, Y_merged 
    """
    # initial X
    X_merged = X_train.copy()
    # assign Y using one hot encodeing 
    Y_merged = np.zeros((X_train.shape[0], len(FLAGS.attack_type) + 1))
    Y_merged[:,0] = np.ones(X_train.shape[0])
    for i, attack_name in enumerate(FLAGS.attack_type):
        # generate adversrial X
        X_train_adv = do_transform(sess, x, attack_dict[attack_name](x), X_train, args=eval_params)
        # constructe Y
        Y_train_adv = np.zeros((X_train.shape[0], len(FLAGS.attack_type) + 1))
        Y_train_adv[:,i+1] = np.ones(X_train.shape[0])
        # merge them to X_merged and Y_merged
        X_merged = np.vstack((X_merged, X_train_adv))
        Y_merged = np.vstack((Y_merged, Y_train_adv))
    
    return X_merged, Y_merged


def get_model(dataset, attack_model, scope, nb_classes, nb_filters, input_shape):
    print(dataset, attack_model)
    if dataset == 'mnist':
        if attack_model == 'basic_model':
            model = ModelBasicCNN(scope, nb_classes, nb_filters)
        elif attack_model == 'mnist_model':
            model = MadryMNIST()
    elif dataset == 'cifar10':
        if attack_model == 'all_cnn':
            model = ModelAllConvolutional(
                scope, nb_classes, nb_filters, input_shape=input_shape)
        elif attack_model == 'cifar10_model':
            model = make_wresnet(scope=scope)
    else:
        raise ValueError(dataset)
    return model


def get_attack(attack_type, model, sess):
    if attack_type == 'fgsm':
        attack = FastGradientMethod(model)
    elif attack_type == 'bim':
        attack = BasicIterativeMethod(model)
    elif attack_type == 'pgd':
        attack = ProjectedGradientDescent(model)
    elif attack_type == 'cwl2':
        attack = CarliniWagnerL2(model, sess)
    elif attack_type == 'jsma':
        attack = SaliencyMapMethod(model)
    else:
        raise ValueError(attack_type)
    return attack


def get_para(dataset, attack_type):
    if dataset == 'mnist' or 'fmnist':
        print('attack_type', attack_type)
        attack_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}
        if attack_type == 'fgsm':
            attack_params = attack_params
        elif attack_type == 'bim':
            attack_params.update({'nb_iter': 50, 'eps_iter': .01})
        elif attack_type == 'pgd':
            attack_params.update({'eps_iter': 0.05, 'nb_iter': 20})
        elif attack_type == 'cwl2':
            attack_params = {'binary_search_steps': 1,
                             'max_iterations': 100,
                             'learning_rate': 0.2,
                             'batch_size': 8,
                             'initial_const': 10}
        elif attack_type == 'jsma':
            attack_params.update({'theta': 1., 'gamma': 0.1,
                                  'clip_min': 0., 'clip_max': 1.,
                                  'y_target': None})
        else:
            raise ValueError(attack_type)

    elif dataset == 'cifar10':
        attack_params = {'clip_min': 0., 'clip_max': 255.}
        if attack_type == 'cwl2':
            attack_params.update({'binary_search_steps': 1,
                                  'max_iterations': 100,
                                  'learning_rate': 0.2,
                                  'batch_size': 8,
                                  'initial_const': 10})
        else:
            attack_params.update({'eps': 8, 'ord': np.inf})
            if attack_type == 'fgsm':
                attack_params = attack_params
            elif attack_type == 'bim':
                attack_params.update({'nb_iter': 50, 'eps_iter': .01})
            elif attack_type == 'pgd':
                attack_params.update({'eps_iter': 2, 'nb_iter': 20})
            elif attack_type == 'jsma':
                attack_params.update({'theta': 1., 'gamma': 0.1,
                                      'clip_min': 0., 'clip_max': 1.,
                                      'y_target': None})
            else:
                raise ValueError(attack_type)
    return attack_params


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

    flags.DEFINE_list('attack_type', ['fgsm'], ("Attack type: 'fgsm'->'fast gradient sign method', "
                                                       "'pgd'->'projected gradient descent', "
                                                       "'bim'->'basic iterative method',"
                                                       "'cwl2'->'Carlini & Wagner L2',"
                                                       "'jsma'->'jsma method'"))
    flags.DEFINE_string('dataset', 'mnist',
                        ("dataset: 'mnist'->'mnist dataset', "
                         "'fmnist'->'fashion mnist dataset', "
                         "'cifar10'->'cifar-10 dataset'"))
    flags.DEFINE_string('attack_model', 'mnist_model',
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
