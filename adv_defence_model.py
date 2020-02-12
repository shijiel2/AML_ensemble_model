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
from cleverhans.train import train
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10
from cleverhans.model import CallableModelWrapper
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
#from utils_fmnist import data_fmnist
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.loss import CrossEntropy
from sklearn.model_selection import train_test_split
from utils import get_model, get_attack, get_para, get_merged_train_data, do_sess_batched_eval, do_preds, write_exp_summary, do_eval, test_detector

FLAGS = flags.FLAGS


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
TRAIN_SIZE = 60000
TEST_SIZE = 10000
OUTPUT_FILE = './output.txt'
EVAL_DETECTOR = False
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
    set_log_level(logging.INFO)

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
    print('Train clean model_0')
    train(sess, loss, X_train, Y_train,
          args=train_params, rng=rng, var_list=model.get_params())

    # For the 1...N attack methods, create adv samples and train defence models model_1...model_n
    for i, attack_name in enumerate(FLAGS.attack_type):
        attack_params = get_para(FLAGS.dataset, attack_name)
        # special case for PGD trainning
        if attack_name == 'pgd':
            attack_params.update({'nb_iter': 10})

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
        print('Train defence model_i for attack:', attack_name)
        train(sess, loss_i, X_train, Y_train,
              args=train_params, rng=rng, var_list=model_i.get_params())

        def_model_list.append(model_i)

    # Make and train detector that classfies whcih source (clean or adv_1 or adv_2...)
    # prepare data
    print('Preparing merged data...')
    X_merged, Y_merged = get_merged_train_data(
        sess, x, attack_dict, X_train, eval_params, FLAGS.attack_type)
    # train the detector
    if FLAGS.dataset == 'mnist':
        detector = ModelBasicCNN(
            'detector', (len(FLAGS.attack_type) + 1), nb_filters)
    elif FLAGS.dataset == 'cifar10':
        detector = ModelAllConvolutional(
            'detector', (len(FLAGS.attack_type) + 1), nb_filters, input_shape=input_shape)

    loss_d = CrossEntropy(detector, smoothing=label_smoothing)
    print('Train detector with X_merged and Y_merged shape:',
          X_merged.shape, Y_merged.shape)
    

    # eval detector
    if EVAL_DETECTOR:
        X_merged_train, X_merged_test, Y_merged_train, Y_merged_test = train_test_split(
            X_merged, Y_merged, test_size=0.20, random_state=42)
        train(sess, loss_d, X_merged_train, Y_merged_train,
              args=train_params, rng=rng, var_list=detector.get_params())
        do_eval(sess, x, tf.placeholder(tf.float32, shape=(None, (len(FLAGS.attack_type) + 1))),
                do_preds(x, detector, 'probs'), X_merged_test, Y_merged_test, 'Detector accuracy', eval_params)
    else:
        train(sess, loss_d, X_merged, Y_merged,
              args=train_params, rng=rng, var_list=detector.get_params())

    # Make Ensemble model
    def ensemble_model_logits(x):
        return do_preds(x, model, 'logits', def_model_list=def_model_list, detector=detector)

    def ensemble_model_logits_unweighted(x):
        return do_preds(x, model, 'logits', def_model_list=def_model_list)

    def ensemble_model_probs(x):
        return tf.math.log(do_preds(x, model, 'probs', def_model_list=def_model_list, detector=detector))

    def ensemble_model_probs_unweighted(x):
        return tf.math.log(do_preds(x, model, 'probs', def_model_list=def_model_list))

    ensemble_model_L = CallableModelWrapper(ensemble_model_logits, 'logits')
    ensemble_model_L_U = CallableModelWrapper(ensemble_model_logits_unweighted, 'logits')
    ensemble_model_P = CallableModelWrapper(ensemble_model_probs, 'logits')
    ensemble_model_P_U = CallableModelWrapper(ensemble_model_probs_unweighted, 'logits')

    # Evaluate the accuracy of model on clean examples
    file_name = FLAGS.attack_model + '_' + str(IS_ONLINE) + '.txt'
    fp = open(file_name, 'w')
    write_exp_summary(fp, FLAGS, IS_ONLINE)
    fp.write('\n\n=============== Results on clean data ===============\n\n')

    print('Evaluating on clean data...')
    do_eval(sess, x, y, do_preds(x, model, 'probs'),
            X_test, Y_test, "origin model on clean data", eval_params, fp)
    do_eval(sess, x, y, do_preds(x, ensemble_model_P, 'probs'),
            X_test, Y_test, "probs ensemble model on clean data", eval_params, fp)
    do_eval(sess, x, y, do_preds(x, ensemble_model_L, 'probs'),
            X_test, Y_test, "logits ensemble model on clean data", eval_params, fp)
    
    # test performance of detector
    test_detector(sess, x, detector, x, X_test, (X_test.shape[0], len(FLAGS.attack_type)+1), eval_params, fp=fp)


    # Evaluate the accuracy of model on adv examples
    for attack_name in FLAGS.attack_type:
        print('Evaluating on attack ' + attack_name + '...')
        fp.write('\n\n=============== Results on attack ' + attack_name + ' ===============\n\n')
        attack_params = get_para(FLAGS.dataset, attack_name)

        model_name_dict = {
            model: 'model_0',
            ensemble_model_L: 'weighted logits ensemble model',
            ensemble_model_P: 'weighted probs ensemble model',
            ensemble_model_L_U: 'unweighted logits ensemble model',
            ensemble_model_P_U: 'unweighted probs ensemble model'
        }

        def attack_from_to(from_model, to_model_lst):
            from_attack = get_attack(attack_name, from_model, sess)
            adv_x = from_attack.generate(x, **attack_params)

            # test performance of detector
            test_detector(sess, x, detector, adv_x, X_test, (X_test.shape[0], len(FLAGS.attack_type)+1), eval_params, fp=fp)

            for to_model in to_model_lst:
                start = time.time()
                message = "==ATTACK ON=> " + model_name_dict[from_model] + " ==TEST ON=> " + model_name_dict[to_model]
                do_eval(sess, x, y, do_preds(adv_x, to_model, 'probs'), X_test, Y_test,
                        message, eval_params, fp)
                end = time.time()
                print('   ' + message + ' Used: ' + str(end - start))
            fp.write('\n')


        # generate attack to origin model
        attack_from_to(model, [model, ensemble_model_L, ensemble_model_P])

        # generate attack to weighted ensemble model
        # logits
        attack_from_to(ensemble_model_L, [model, ensemble_model_L])
        # probs
        attack_from_to(ensemble_model_P, [model, ensemble_model_P])

        # generate attack to unweighted ensemble model
        # logits
        attack_from_to(ensemble_model_L_U, [model, ensemble_model_L])
        # probs
        attack_from_to(ensemble_model_P_U, [model, ensemble_model_P])

    fp.close()

        



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

    flags.DEFINE_list('attack_type', ['fgsm', 'pgd'], ("Attack type: 'fgsm'->'fast gradient sign method', "
                                                       "'pgd'->'projected gradient descent', "
                                                       "'bim'->'basic iterative method',"
                                                       "'cwl2'->'Carlini & Wagner L2',"
                                                       "'jsma'->'jsma method'"))
    flags.DEFINE_string('dataset', 'mnist',
                        ("dataset: 'mnist'->'mnist dataset', "
                         "'fmnist'->'fashion mnist dataset', "
                         "'cifar10'->'cifar-10 dataset'"))
    flags.DEFINE_string('attack_model', 'basic_model',
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
