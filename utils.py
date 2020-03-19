import tensorflow as tf
from cleverhans.utils_tf import model_eval
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
import numpy as np
import math
from tqdm import tqdm
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
from models.cifar10_model import make_wresnet
from models.mnist_model import MadryMNIST
from models.simple_model import SimpleLinear
from models.simple_Cifar10 import SimpleCifar10
from cleverhans.attacks import *
import datetime
from sklearn.model_selection import train_test_split
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from settings import Settings


"""
**************************** Helper Functions ****************************
"""


def get_model(scope, nb_classes=Settings.nb_classes, nb_filters=Settings.NB_FILTERS, input_shape=Settings.input_shape):
    dataset = Settings.dataset
    attack_model = Settings.attack_model

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
    elif attack_type == 'spsa':
        attack = SPSA(model)
    else:
        raise ValueError(attack_type)
    return attack


def get_para(attack_type):
    dataset = Settings.dataset

    if dataset == 'mnist' or 'fmnist':
        attack_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}
        if attack_type == 'fgsm':
            attack_params = attack_params
        elif attack_type == 'spsa':
            attack_params.update({'nb_iter': 10})
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


def do_eval(sess, pred, X_test, Y_test, message, args):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param pred: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                Should contain `batch_size`
    """
    x = Settings.x
    y = Settings.y
    acc = model_eval(
        sess, x, y, pred, X_test, Y_test, args=args)
    Settings.fp.write(message + '\nAccuracy: %0.4f\n' % acc)
    print(message + '\nAccuracy: %0.4f\n' % acc)


def do_preds(x, model, method, def_model_list=None, detector=None):
    """
    Return the mean prob of multiple models
    :param x: input 
    :param model: model used to make prediction
    :param method: specify output type 'logits' or 'probs'
    :param def_model_list: a list of models, used to construct ensemble model, if not provided only use model
    :param detector: assign different weights to each model in def_model_list, if not provided using same weights
    :return logits or probs predictions for input x 
    """
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    if detector is None:
        preds = tf.stack([get_pred(x, m, method) for m in models], 1)
        return tf.reduce_mean(preds, 1)
    else:
        weights = get_pred(x, detector, 'probs')
        preds = tf.stack([get_pred(x, m, method) for m in models], 1)
        preds = tf.math.multiply(preds, weights[:, :, None])
        return tf.reduce_sum(preds, 1)


def get_pred(x, model, method):
    """
    Wrapper function of get_probs() and get_logtits(), embedding with randomness. If randomness is true in Settings, noises 
    will be added to the input x to construct k different inputs, the output is the reduce mean of their outputs. 
    :param x: input 
    :param model: model used to make prediction
    :param method: specify output type 'logits' or 'probs'
    :return logits or probs predictions for input x 
    """
    if method == 'probs':
        if not Settings.PRED_RANDOM:
            return model.get_probs(x)
        else:
            probs = []
            for _ in range(Settings.RANDOM_K):
                new_x = x + \
                    tf.random.normal(tf.shape(x), dtype=x.dtype,
                                     stddev=Settings.RANDOM_STDDEV)
                probs.append(model.get_probs(new_x))
            return tf.reduce_mean(tf.stack(probs, 0), 0)

    elif method == 'logits':
        return model.get_logits(x)


def do_sess_batched_eval(sess, x_gen, X_in, Y_in, X_out_shape, args):
    """
    Apply gen to ndarray, and return a new ndarray.
    :param sess: TF session to use
    :prarm x: placeholder for data in X_in
    :param x_gen: generator feed with placeholder, eg. attack(x)
    :param X_in: numpy array with inputs
    :return: ndarry with same shape 
    """
    args = _ArgsWrapper(args or {})
    assert args.batch_size, "Batch size was not given in args dict"
    if X_in is None:
        raise ValueError("X_in argument must be supplied.")

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_in)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_in)
        # final output ndarray
        X_out = np.zeros(X_out_shape, dtype=X_in.dtype)

        for batch in tqdm(range(nb_batches)):
            start = batch * args.batch_size
            end = min(len(X_in), start + args.batch_size)
            feed_dict = {
                Settings.x: X_in[start:end],
                Settings.y: Y_in[start:end]
            }
            X_out[start:end] = x_gen.eval(feed_dict=feed_dict)

        assert end >= len(X_in)

    return X_out


def get_merged_train_data(sess, attack_dict, attack_types, X_train, Y_train):
    """
    Construct the merged data for trainning the classifier
    :param sess: current session
    :param attack_dict: dictionary {attack_name: attack_generator}
    :param X_train: ndarray 
    :return: X_merged, Y_merged 
    """
    # initial X
    X_merged = X_train.copy()
    # assign Y using one hot encodeing
    Y_merged = np.zeros((X_train.shape[0], len(attack_types) + 1))
    Y_merged[:, 0] = np.ones(X_train.shape[0])
    for i, attack_name in enumerate(attack_types):
        # generate adversrial X
        if 'spsa' in attack_name:
            args = {'batch_size': 1}
        else:
            args = {'batch_size': Settings.BATCH_SIZE}
        X_train_adv = do_sess_batched_eval(
            sess, attack_dict[attack_name](Settings.x), X_train, Y_train, X_train.shape, args=args)
        # constructe Y
        Y_train_adv = np.zeros((X_train.shape[0], len(attack_types) + 1))
        Y_train_adv[:, i+1] = np.ones(X_train.shape[0])
        # merge them to X_merged and Y_merged
        X_merged = np.vstack((X_merged, X_train_adv))
        Y_merged = np.vstack((Y_merged, Y_train_adv))

    return X_merged, Y_merged


def get_detector(name, width):
    """
    Fetch the corresponding detector.
    :param name: name of the detector 
    :param width: how many catagories the detector could classify
    :return: detector
    """

    if Settings.dataset == 'mnist':
        if not Settings.LINEAR_DETECTOR:
            detector = ModelBasicCNN(name, width, Settings.NB_FILTERS)
        else:
            detector = SimpleLinear(name, width, Settings.NB_FILTERS)

    elif Settings.dataset == 'cifar10':
        if not Settings.LINEAR_DETECTOR:
            detector = ModelAllConvolutional(
                name, width, Settings.NB_FILTERS, input_shape=Settings.input_shape)
        else:
            detector = SimpleCifar10(
                name, width, Settings.NB_FILTERS, input_shape=Settings.input_shape)

    return detector


def test_detector(sess, detector, x_in, X_test, Y_test, X_out_shape, args):
    """
    Test the performance of detector by reproducing the weights it assigned in each adv examples.
    :param sess: current session
    :param detector: the detector to be tested
    :param x_in: the adv examples
    :param X_test: ndarray stroing actual data
    :param X_out_shape: shape of output ndarray
    """
    det_probs = get_pred(x_in, detector, 'probs')
    Probs = do_sess_batched_eval(
        sess, det_probs, X_test, Y_test, X_out_shape, args=args)

    Preds = np.argmax(Probs, axis=1)
    unique, counts = np.unique(Preds, return_counts=True)

    Settings.fp.write('detector mean probs:' +
                      str(np.mean(Probs, axis=0)) + '\n')
    Settings.fp.write('detector preds counts:' +
                      str(dict(zip(unique, counts))) + '\n')

    print('detector mean probs:' + str(np.mean(Probs, axis=0)) + '\n')
    print('detector preds counts:' + str(dict(zip(unique, counts))))


def train_detector(sess, detector, attack_dict, attack_types, X_train, Y_train):
    """
    Train a detector and return it.
    :param sess: current session
    :param x: placeholder for trainning data
    :param detector: the detector
    :param attack_dict: dictionary {attack_name: attack_generator}
    :param attack_types: list of attacks it need to be detect
    :param X_train: ndarray 
    """
    # Make and train detector that classfies whcih source (clean or adv_1 or adv_2...)
    # prepare data
    print('Preparing merged data...')
    X_merged, Y_merged = get_merged_train_data(
        sess, attack_dict, attack_types, X_train, Y_train)

    loss_d = CrossEntropy(detector, smoothing=Settings.LABEL_SMOOTHING)
    print('Train detector with X_merged and Y_merged shape:',
          X_merged.shape, Y_merged.shape)

    train(sess, loss_d, X_merged, Y_merged,
          args=Settings.train_params, rng=Settings.rng, var_list=detector.get_params())

    return detector


def get_attack_fun(from_model, attack_name, sess):
    attack_params = get_para(attack_name)
    attack_method = get_attack(attack_name, from_model, sess)

    def attack(x):
        if 'spsa' in attack_name:
            return attack_method.generate(x, y=Settings.y, **attack_params)
        else:
            return attack_method.generate(x, **attack_params)
    return attack


def train_defence_model(sess, model_i, from_model, attack_name, X_train, Y_train):
    attack_method = get_attack(attack_name, from_model, sess)
    attack_params = get_para(attack_name)
    train_params = Settings.train_params
    pass_y = False
    if 'spsa' in attack_name:
        train_params['batch_size'] = 1
        pass_y = True

    print('Trainnig model:', attack_name)
    loss_i = CrossEntropy(
        model_i, smoothing=Settings.LABEL_SMOOTHING, attack=attack_method, adv_coeff=1., attack_params=attack_params, pass_y=pass_y)
    train(sess, loss_i, X_train, Y_train,
          args=train_params, rng=Settings.rng, var_list=model_i.get_params())


def reinfrocement_ensemble_gen(sess, name, from_model, def_model_list, attack_dict, X_train, Y_train):
    rein_def_model_list = def_model_list.copy()

    for attack_name in Settings.REINFORE_ENS:
        model_i = get_model(name + '_rein_def_model_' + attack_name)
        train_defence_model(sess, model_i, from_model,
                            attack_name, X_train, Y_train)

        rein_def_model_list.append(model_i)
        attack_dict[str(name + attack_name)
                    ] = get_attack_fun(from_model, attack_name, sess)

    rein_attack_types = Settings.attack_type + \
        [str(name + a) for a in Settings.REINFORE_ENS]
    rein_detector = get_detector(
        'rein_detector_'+name, len(rein_def_model_list)+1)
    train_detector(sess, rein_detector, attack_dict,
                   rein_attack_types, X_train, Y_train)

    return rein_def_model_list, rein_detector


def write_exp_summary():
    fp = Settings.fp
    string = 'Experiment for AML project.\n' + \
             'Date: ' + str(datetime.datetime.now()) + '\n\n' + \
             'Dataset: ' + Settings.dataset + '\n' + \
             'Attack types: ' + str(Settings.attack_type) + '\n' + \
             'Eval attack types: ' + str(Settings.eval_attack_type) + '\n' + \
             'Model type: ' + str(Settings.attack_model) + '\n' + \
             'Is Online: ' + str(Settings.IS_ONLINE) + '\n' + \
             'Randomness in prediction: ' + str(Settings.PRED_RANDOM) + '\n' + \
             'Random K: ' + str(Settings.RANDOM_K) + '\n' + \
             'Stddev: ' + str(Settings.RANDOM_STDDEV) + '\n' + \
             'Ensemble reinforcement: ' + str(Settings.REINFORE_ENS) + '\n' + \
             'Linear detector: ' + str(Settings.LINEAR_DETECTOR) + '\n' + \
             '\n\n'
    fp.write(string)
