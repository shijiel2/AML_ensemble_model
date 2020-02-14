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
from cleverhans.train import train
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
from sklearn.model_selection import train_test_split
from get_model import *

FLAGS = flags.FLAGS


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
TRAIN_SIZE = 60000
TEST_SIZE = 10000
IS_ONLINE = True
CHECK_DET = False
NOISE = True


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

    adv_x = {}
    # define and train clean model to be defenced on
    model = get_model(FLAGS.dataset, FLAGS.attack_model, 'model', nb_classes, nb_filters,
                      input_shape)
    rng = np.random.RandomState([2017, 10, 30])
    loss = CrossEntropy(model, smoothing=label_smoothing)
    train(sess, loss, X_train, Y_train,
          args=train_params, rng=rng, var_list=model.get_params())
    adv_x['model0'] = x

    # for the 1...M attack methods, create adv samples and train defence models
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
            
        loss_i = CrossEntropy(
            model_i, smoothing=label_smoothing, attack=attack, adv_coeff=1.)
        train(sess, loss_i, X_train, Y_train,
              args=train_params, rng=rng, var_list=model_i.get_params())

        def_model_list.append(model_i)
        adv_x[attack_name] = attack(x)


    '''
    --------------- weighted class train --------------
    '''

    # define and train attack detector model
    new_nb_classes = len(adv_x)
    # build a CNN model for the detector
    # det_model = ModelBasicCNN('detector', new_nb_classes, nb_filters)

    # build a simple linear model for the detector
    det_model = SimpleLinear('detector', new_nb_classes, nb_filters)
    train_attack_detector(sess,x,det_model,adv_x, X_train, Y_train, train_params_det, rng)

    '''
    ----------------- create ens model ---------------
    '''
    # Make Ensemble model
    def ensemble_model_logits(x):
        return do_logits(x, model, def_model_list=def_model_list)
    def ensemble_model_probs(x):
        return tf.math.log(do_probs(x, model, def_model_list=def_model_list))

    ensemble_model_L = CallableModelWrapper(ensemble_model_logits, 'logits')
    ensemble_model_P = CallableModelWrapper(ensemble_model_probs, 'logits')


    # Make Ensemble model
    def weighted_ensemble_model_logits(x):
        return do_logits(x, model,det_model, def_model_list=def_model_list)
    def weighted_ensemble_model_probs(x):
        return tf.math.log(do_probs(x, model,det_model, def_model_list=def_model_list))

    weighted_ensemble_model_L = CallableModelWrapper(weighted_ensemble_model_logits, 'logits')
    weighted_ensemble_model_P = CallableModelWrapper(weighted_ensemble_model_probs, 'logits')


    '''
    ----------------- model eval ----------------
    '''
    # Evaluate the accuracy of model on clean examples
    with sess.as_default():
        print(tf.reduce_mean(det_model.get_probs(x),0).eval(feed_dict={x:X_test}))
    do_eval(sess, x, y, do_probs(x, model), 
            X_test, Y_test, "origin model on clean data", eval_params)
    do_eval(sess, x, y, do_probs(x, ensemble_model_L), 
            X_test, Y_test, "ensemble model on clean data, with logits", eval_params)
    do_eval(sess, x, y, do_probs(x, ensemble_model_P), 
            X_test, Y_test, "ensemble model on clean data, with probs", eval_params)
    do_eval(sess, x, y, do_probs(x, weighted_ensemble_model_L), 
            X_test, Y_test, "weighted ensemble model on clean data, with logits", eval_params)
    do_eval(sess, x, y, do_probs(x, weighted_ensemble_model_P), 
            X_test, Y_test, "weighted ensemble model on clean data, with probs", eval_params)    

    # Evaluate the accuracy of model on adv examples
    for i, attack_name in enumerate(FLAGS.attack_type):
        attack_params = get_para(FLAGS.dataset, attack_name, att=True)

        '''---------------------origin model case----------------------'''
        # generate attack to origin model
        origin_attack = get_attack(attack_name, model, sess)
        origin_adv_x = origin_attack.generate(x, **attack_params)
        
        # check the weights from the detector for model0
        with sess.as_default():
            print(attack_name, 'model0', 
                  tf.reduce_mean(det_model.get_probs(origin_adv_x),0).eval(feed_dict={x:X_test}))
        do_eval(sess, x, y, do_probs(origin_adv_x, model), 
                X_test, Y_test, attack_name + "-> origin model, test on origin model", eval_params)
        do_eval(sess, x, y, do_probs(origin_adv_x, ensemble_model_L), 
                X_test, Y_test, attack_name + "-> origin model, test on ensemble model, using logits", eval_params)
        do_eval(sess, x, y, do_probs(origin_adv_x, ensemble_model_P), 
                X_test, Y_test, attack_name + "-> origin model, test on ensemble model, using probs", eval_params)
        do_eval(sess, x, y, do_probs(origin_adv_x, weighted_ensemble_model_L), 
                X_test, Y_test, attack_name + "-> origin model, test on weighted ensemble model, using logits", eval_params)
        do_eval(sess, x, y, do_probs(origin_adv_x, weighted_ensemble_model_P), 
                X_test, Y_test, attack_name + "-> origin model, test on weighted ensemble model, using probs", eval_params)
        
        
        '''---------------------online model case----------------------'''     
        if IS_ONLINE == True:
            # modeli
            modeli_attack = get_attack(attack_name, def_model_list[i], sess)
            modeli_adv_x = modeli_attack.generate(x, **attack_params)

            # check the weights from the detector for modeli
            with sess.as_default():
                print(attack_name, 'modeli', 
                      tf.reduce_mean(det_model.get_probs(modeli_adv_x),0).eval(feed_dict={x:X_test}))
            do_eval(sess, x, y, do_probs(modeli_adv_x, model), 
                    X_test, Y_test, attack_name + "-> online modeli, test on origin model", eval_params)
            do_eval(sess, x, y, do_probs(modeli_adv_x, ensemble_model_L), 
                    X_test, Y_test, attack_name + "-> online modeli, test on ensemble model, using logits", eval_params)
            do_eval(sess, x, y, do_probs(modeli_adv_x, ensemble_model_P), 
                    X_test, Y_test, attack_name + "-> online modeli, test on ensemble model, using probs", eval_params)
            do_eval(sess, x, y, do_probs(modeli_adv_x, weighted_ensemble_model_L), 
                    X_test, Y_test, attack_name + "-> online model, test on weighted ensemble model, using logits", eval_params)
            do_eval(sess, x, y, do_probs(modeli_adv_x, weighted_ensemble_model_P), 
                    X_test, Y_test, attack_name + "-> online model, test on weighted ensemble model, using probs", eval_params)
                            
        '''---------------------ensemble model case----------------------'''
        # generate attack from un-w ensemble model attack wgted, based on probs
        print('based on probs')
        ensemble_attack_P = get_attack(attack_name, ensemble_model_P, sess)
        ensemble_adv_x_P = ensemble_attack_P.generate(x, **attack_params)
        weighted_ensemble_attack_P = get_attack(attack_name, weighted_ensemble_model_P, sess)
        weighted_ensemble_adv_x_P = weighted_ensemble_attack_P.generate(x, **attack_params)
        
        # check the weights from the detector for ens models
        with sess.as_default():
            print(attack_name, 'unweighted_ens', 
                  tf.reduce_mean(det_model.get_probs(ensemble_adv_x_P),0).eval(feed_dict={x:X_test}))
            print(attack_name, 'weighted_ens', 
                  tf.reduce_mean(det_model.get_probs(weighted_ensemble_adv_x_P),0).eval(feed_dict={x:X_test}))
        
        do_eval(sess, x, y, do_probs(ensemble_adv_x_P, model), 
            X_test, Y_test, attack_name + "-> ensemble model, test on original model", eval_params)
        do_eval(sess, x, y, do_probs(ensemble_adv_x_P, ensemble_model_P), 
            X_test, Y_test, attack_name + "-> ensemble model, test on ensemble model", eval_params)
        do_eval(sess, x, y, do_probs(weighted_ensemble_adv_x_P, model), 
            X_test, Y_test, attack_name + "-> weighted_ensemble model, test on original model", eval_params)
        do_eval(sess, x, y, do_probs(weighted_ensemble_adv_x_P, weighted_ensemble_model_P), 
            X_test, Y_test, attack_name + "-> weighted_ensemble model, test on weighted_ensemble model", eval_params)       
        do_eval(sess, x, y, do_probs(ensemble_adv_x_P, weighted_ensemble_model_P), 
            X_test, Y_test, attack_name + "-> from unweighted, on weighted", eval_params)

        # generate attack from un-w ensemble model attack wgted, based on logits
        print('based on logits')
        ensemble_attack_L = get_attack(attack_name, ensemble_model_L, sess)
        ensemble_adv_x_L = ensemble_attack_P.generate(x, **attack_params)
        weighted_ensemble_attack_L = get_attack(attack_name, weighted_ensemble_model_L, sess)
        weighted_ensemble_adv_x_L = weighted_ensemble_attack_L.generate(x, **attack_params)

        do_eval(sess, x, y, do_probs(ensemble_adv_x_L, model), 
            X_test, Y_test, attack_name + "-> ensemble model, test on original model", eval_params)
        do_eval(sess, x, y, do_probs(ensemble_adv_x_L, ensemble_model_L), 
            X_test, Y_test, attack_name + "-> ensemble model, test on ensemble model", eval_params)
        do_eval(sess, x, y, do_probs(weighted_ensemble_adv_x_L, model), 
            X_test, Y_test, attack_name + "-> weighted_ensemble model, test on original model", eval_params)
        do_eval(sess, x, y, do_probs(weighted_ensemble_adv_x_L, weighted_ensemble_model_L), 
            X_test, Y_test, attack_name + "-> weighted_ensemble model, test on weighted_ensemble model", eval_params)       
        do_eval(sess, x, y, do_probs(ensemble_adv_x_L, weighted_ensemble_model_L),
                X_test, Y_test, attack_name + "-> from unweighted, on weighted", eval_params)

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
    nb_noise = 5
    noise_x = []
    for i in range(nb_noise):
        noise_x.append(x+tf.random_normal(shape = tf.shape(x), 
                                mean=0.0, stddev=0.1, dtype=tf.float32))
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
