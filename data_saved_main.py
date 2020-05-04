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
from data_saved_utils import get_attack, get_attack_fun, get_detector, get_model, get_para, \
    train_defence_model, do_preds, reinfrocement_ensemble_gen, train_detector, \
    test_detector, do_eval, write_exp_summary, save_data, load_data, make_adv_data, train_defence_model_online


def defence_frame():
    """
    **************************** Settings & Parameters ****************************
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.WARN)

    # Create TF session
    sess = tf.Session(config=tf.ConfigProto(**Settings.config_args))

    # list of model_1...model_n
    def_model_list = []
    # dict of attack_name -> attack function
    attack_dict = {}
    # dict of model names
    model_name_dict = {}

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
        model_i = get_model('basic_' + attack_name)
        if Settings.IS_ONLINE:
            from_model = model_i
        else:
            from_model = model

        train_defence_model(sess, model_i, from_model, attack_name, X_train, Y_train)
        
        attack_dict[attack_name] = get_attack_fun(
            from_model, attack_name, sess)
        def_model_list.append(model_i)

    # The m+1 adv_training model to defend ensemble model itself
    self_defence_model_logits = get_model('model_self_defence_logits')
    model_name_dict[self_defence_model_logits] = 'model self defence logits'

    self_defence_model_probs = get_model('model_self_defence_probs')
    model_name_dict[self_defence_model_probs] = 'model self defence probs'

    # Make unweighted Ensemble model
    def ensemble_model_logits_unweighted(x):
        return do_preds(x, model, 'logits', def_model_list=def_model_list)

    def ensemble_model_probs_unweighted(x):
        return tf.math.log(do_preds(x, model, 'probs', def_model_list=def_model_list))

    # unweighted ensemble: model_0, model_fgsm, model_pgd
    ensemble_model_L_U = CallableModelWrapper(ensemble_model_logits_unweighted, 'logits')
    model_name_dict[ensemble_model_L_U] = 'unweighted logits ensemble model'

    ensemble_model_P_U = CallableModelWrapper(ensemble_model_probs_unweighted, 'logits')
    model_name_dict[ensemble_model_P_U] = 'unweighted probs ensemble model'

    # Make blackbox samples data if needed
    if load_data(Settings.blackbox_samples) is None:
        save_data(make_adv_data(sess, ensemble_model_L_U, Settings.BLACKBOX_SAMPLES_METHOD, X_train, Y_train), Settings.blackbox_samples)

    # Reinforce weighted ensemble model
    if Settings.REINFORE_ENS:

        logits_ens_def_model_list, logits_detector = reinfrocement_ensemble_gen(
            sess, 'logits', ensemble_model_L_U, def_model_list, attack_dict, X_train, Y_train)
        probs_ens_def_model_list, probs_detector = reinfrocement_ensemble_gen(
            sess, 'probs', ensemble_model_P_U, def_model_list, attack_dict, X_train, Y_train)

        def ensemble_model_logits(x):
            return do_preds(x, model, 'logits', def_model_list=logits_ens_def_model_list + [self_defence_model_logits], detector=logits_detector)

        def ensemble_model_probs(x):
            return tf.math.log(do_preds(x, model, 'probs', def_model_list=probs_ens_def_model_list + [self_defence_model_probs], detector=probs_detector))

        def ensemble_model_logits_unweighted_alter(x):
            return do_preds(x, model, 'logits', def_model_list=logits_ens_def_model_list)

        def ensemble_model_probs_unweighted_alter(x):
            return tf.math.log(do_preds(x, model, 'probs', def_model_list=probs_ens_def_model_list))

        # weighted ensemble model: model_0, model_fgsm, model_pgd, rein_model_pgd, rein_model_fgsm
        ensemble_model_L = CallableModelWrapper(ensemble_model_logits, 'logits')
        model_name_dict[ensemble_model_L] = 'weighted logits ensemble model'

        ensemble_model_P = CallableModelWrapper(ensemble_model_probs, 'logits')
        model_name_dict[ensemble_model_P] = 'weighted probs ensemble model'

        # unweighted ensemble model: model_0, model_fgsm, model_pgd, rein_model_pgd, rein_model_fgsm
        ensemble_model_L_U_alter = CallableModelWrapper(ensemble_model_logits_unweighted_alter, 'logits')
        model_name_dict[ensemble_model_L_U_alter] = 'same sized unweighted logits ensemble model'

        ensemble_model_P_U_alter = CallableModelWrapper(ensemble_model_probs_unweighted_alter, 'logits')
        model_name_dict[ensemble_model_P_U_alter] = 'same sized unweighted probs ensemble model'

    else:
        detector = get_detector('detector_general', len(def_model_list) + Settings.def_list_addon)
        train_detector(sess, detector, def_model_list, X_train, Y_train)

        def ensemble_model_logits(x):
            return do_preds(x, model, 'logits', def_model_list=def_model_list + [self_defence_model_logits], detector=detector)

        def ensemble_model_probs(x):
            return tf.math.log(do_preds(x, model, 'probs', def_model_list=def_model_list + [self_defence_model_probs], detector=detector))

        # weighted ensemble model: model_0, model_fgsm, model_pgd
        ensemble_model_L = CallableModelWrapper(ensemble_model_logits, 'logits')
        model_name_dict[ensemble_model_L] = 'weighted logits ensemble model'

        ensemble_model_P = CallableModelWrapper(ensemble_model_probs, 'logits')
        model_name_dict[ensemble_model_P] = 'weighted probs ensemble model'

    # Train self defence model
    train_defence_model_online(sess, self_defence_model_logits, ensemble_model_L, 'pgd', X_train, Y_train)
    train_defence_model_online(sess, self_defence_model_probs, ensemble_model_P, 'pgd', X_train, Y_train)

    # Evaluate the accuracy of model on clean examples
    write_exp_summary()
    Settings.fp.write(
        '\n\n=============== Results on clean data ===============\n\n')
    print('Evaluating on clean data...')
    do_eval(sess, do_preds(Settings.x, model, 'probs'),
            X_test, Y_test, "origin model on clean data")
    do_eval(sess, do_preds(Settings.x, ensemble_model_P, 'probs'),
            X_test, Y_test, "probs ensemble model on clean data")
    do_eval(sess, do_preds(Settings.x, ensemble_model_L, 'probs'),
            X_test, Y_test, "logits ensemble model on clean data")

    # test performance of detector
    if Settings.EVAL_DETECTOR:
        if Settings.REINFORE_ENS:
            Settings.fp.write('Detector logits\n')
            test_detector(sess, logits_detector, Settings.x, X_test, Y_test,
                        (X_test.shape[0], len(logits_ens_def_model_list)+2))
            Settings.fp.write('Detector probs\n')
            test_detector(sess, probs_detector, Settings.x, X_test, Y_test,
                        (X_test.shape[0], len(probs_ens_def_model_list)+2))
        else:
            Settings.fp.write('Detector clean\n')
            test_detector(sess, detector, Settings.x, X_test, Y_test,
                        (X_test.shape[0], len(Settings.attack_type)+2))

    # Evaluate the accuracy of model on adv examples
    for attack_name in Settings.eval_attack_type:
        print('Evaluating on attack ' + attack_name + '...')
        Settings.fp.write('\n\n=============== Results on attack ' +
                          attack_name + ' ===============\n\n')

        def attack_from_to(from_model, to_model_lst):
            Adv_X_test_name = attack_name + '_attack_' + from_model.scope + '.npy'
            Adv_X_test = load_data(Adv_X_test_name)
            if load_data(Adv_X_test_name) is None:
                Adv_X_test = make_adv_data(sess, from_model, attack_name, X_test, Y_test)
                save_data(Adv_X_test, Adv_X_test_name)

            # test performance of detector
            if Settings.EVAL_DETECTOR:
                if not Settings.REINFORE_ENS:
                    test_detector(sess, detector, Settings.x, Adv_X_test, Y_test,
                                (Adv_X_test.shape[0], len(Settings.attack_type)+2))
                else:
                    if from_model in (model, ensemble_model_L, ensemble_model_L_U, ensemble_model_L_U_alter):
                        Settings.fp.write('Detector logits\n')
                        test_detector(sess, logits_detector, Settings.x, Adv_X_test, Y_test,
                                    (Adv_X_test.shape[0], len(logits_ens_def_model_list)+2))
                    if from_model in (model, ensemble_model_P, ensemble_model_P_U, ensemble_model_P_U_alter):
                        Settings.fp.write('Detector probs\n')
                        test_detector(sess, probs_detector, Settings.x, Adv_X_test, Y_test,
                                    (Adv_X_test.shape[0], len(probs_ens_def_model_list)+2))

            # eval model accuracy
            for to_model in to_model_lst:
                start = time.time()
                message = "==ATTACK ON=> " + \
                    model_name_dict[from_model] + \
                    " ==TEST ON=> " + model_name_dict[to_model]
                do_eval(sess, do_preds(Settings.x, to_model, 'probs'), Adv_X_test, Y_test, message)
                end = time.time()
                print('   ' + message + ' Used: ' + str(end - start))

            Settings.fp.write('\n')

        # white box attacks
        if attack_name in ['fgsm', 'pgd']:
            # generate attack to origin model
            attack_from_to(model, [model, ensemble_model_L, ensemble_model_P])
            # generate attack from weighted ensemble model
            # logits
            attack_from_to(ensemble_model_L, [ensemble_model_L])
            # probs
            attack_from_to(ensemble_model_P, [ensemble_model_P])

            # generate attack from unweighted ensemble model
            # logits
            attack_from_to(ensemble_model_L_U, [ensemble_model_L])
            # probs
            attack_from_to(ensemble_model_P_U, [ensemble_model_P])

            if Settings.REINFORE_ENS:
                attack_from_to(ensemble_model_L_U_alter, [ensemble_model_L])
                attack_from_to(ensemble_model_P_U_alter, [ensemble_model_P])
        
        # black box attacks
        else:
            attack_from_to(model, [model, ensemble_model_L, ensemble_model_P])
            # logits
            attack_from_to(ensemble_model_L, [ensemble_model_L, self_defence_model_logits])
            # probs
            attack_from_to(ensemble_model_P, [ensemble_model_P, self_defence_model_probs])

    Settings.fp.close()


def main(argv):
    defence_frame()


if __name__ == '__main__':
    app.run(main)
