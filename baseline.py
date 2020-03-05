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
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.train import train, train_ftramer
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10
from cleverhans.model import CallableModelWrapper
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
# from utils_fmnist import data_fmnist
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.loss import CrossEntropy
from sklearn.model_selection import train_test_split
from settings import Settings
from utils import get_attack, get_attack_fun, get_detector, get_model, get_para, \
    train_defence_model, do_preds, reinfrocement_ensemble_gen, train_detector, \
    test_detector, do_eval, write_exp_summary


def defence_frame():
    """
    **************************** Settings & Parameters ****************************
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.ERROR)

    # Create TF session
    sess = tf.Session(config=tf.ConfigProto(**Settings.config_args))

    # list of model_1...model_n
    def_model_list = []
    # dict of attack_name -> attack function
    attack_dict = {}
    # dict of model names
    model_name_dict = {}
    # ftramer baseline attack list
    ftramer_attacks = []

    # Dataset for quick access
    X_train = Settings.X_train
    X_test = Settings.X_test
    Y_train = Settings.Y_train
    Y_test = Settings.Y_test

    """
    **************************** Logic Operations ****************************
    """

    # Define and train model (aka model_0)
    model = get_model('model')
    loss = CrossEntropy(model, smoothing=Settings.LABEL_SMOOTHING)
    print('Train clean model_0')
    train(sess, loss, X_train, Y_train,
          args=Settings.train_params, rng=Settings.rng, var_list=model.get_params())
    model_name_dict[model] = 'model_0'

    # For the 1...N attack methods, create adv samples and train defence models model_1...model_n
    for attack_name in Settings.attack_type:
        model_i = get_model('basic_def_model_' + attack_name)
        if Settings.IS_ONLINE:
            from_model = model_i
        else:
            from_model = model
        train_defence_model(sess, model_i, from_model,
                            attack_name, X_train, Y_train)

        attack_dict[attack_name] = get_attack_fun(
            from_model, attack_name, sess)
        def_model_list.append(model_i)
        ftramer_attacks.append(get_attack(attack_name, from_model, sess))

    # Make baseline: ftramer ensemble model
    ftramer_model = get_model('ftramer_model')
    ftramer_loss = CrossEntropy(ftramer_model, smoothing=Settings.LABEL_SMOOTHING)
    train_ftramer(sess, ftramer_loss, X_train, Y_train, attack_dict,
                  args=Settings.train_params, rng=Settings.rng, var_list=ftramer_model.get_params())
    model_name_dict[ftramer_model] = 'ftramer ensemble model'


    # Evaluate the accuracy of model on clean examples
    write_exp_summary()
    Settings.fp.write(
        '\n\n=============== Results on clean data ===============\n\n')
    print('Evaluating on clean data...')
    do_eval(sess, do_preds(Settings.x, model, 'probs'),
            X_test, Y_test, "origin model on clean data")
    do_eval(sess, do_preds(Settings.x, ftramer_model, 'probs'),
            X_test, Y_test, "ftramer ensemble model on clean data")

    # Evaluate the accuracy of model on adv examples
    for attack_name in Settings.attack_type:
        print('Evaluating on attack ' + attack_name + '...')
        Settings.fp.write('\n\n=============== Results on attack ' +
                          attack_name + ' ===============\n\n')
        attack_params = get_para(attack_name)

        def attack_from_to(from_model, to_model_lst):
            from_attack = get_attack(attack_name, from_model, sess)
            adv_x = from_attack.generate(Settings.x, **attack_params)

            # eval model accuracy
            for to_model in to_model_lst:
                start = time.time()
                message = "==ATTACK ON=> " + \
                    model_name_dict[from_model] + \
                    " ==TEST ON=> " + model_name_dict[to_model]
                do_eval(sess, do_preds(adv_x, to_model, 'probs'), X_test, Y_test,
                        message)
                end = time.time()
                print('   ' + message + ' Used: ' + str(end - start))

            Settings.fp.write('\n')

        # generate attack to origin model
        attack_from_to(model, [model, ftramer_model])
        attack_from_to(ftramer_model, [ftramer_model])

    Settings.fp.close()


def main(argv):
    defence_frame()


if __name__ == '__main__':
    app.run(main)
