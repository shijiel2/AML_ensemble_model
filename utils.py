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
from cleverhans.attacks import *
import datetime
from sklearn.model_selection import train_test_split
from cleverhans.loss import CrossEntropy
from cleverhans.train import train

"""
**************************** Helper Functions ****************************
"""


def get_model(dataset, attack_model, scope, nb_classes, nb_filters, input_shape):
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


def do_eval(sess, x, y, pred, X_test, Y_test, message, eval_params, fp=None):
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
    acc = model_eval(
        sess, x, y, pred, X_test, Y_test, args=eval_params)
    if fp != None:
        fp.write(message + '\nAccuracy: %0.4f\n' % acc)
    else:
        print(message + '\nAccuracy: %0.4f\n' % acc)


def do_preds(x, model, method, def_model_list=None, detector=None, random=False, k=5, stddev=1.0):
    """
    Return the mean prob of multiple models
    """
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    if detector is None:
        preds = tf.stack([get_pred(x, m, method) for m in models], 1)
        return tf.reduce_mean(preds, 1)
    else:
        weights = get_pred(x, detector, 'probs',
                           random=random, k=k, stddev=stddev)
        preds = tf.stack([get_pred(x, m, method) for m in models], 1)
        preds = tf.math.multiply(preds, weights[:, :, None])
        return tf.reduce_sum(preds, 1)


def get_pred(x, model, method, random=False, k=5, stddev=1.0):
    if method == 'probs':
        if not random:
            return model.get_probs(x)
        else:
            probs = []
            for _ in range(k):
                new_x = x + \
                    tf.random.normal(tf.shape(x), dtype=x.dtype, stddev=stddev)
                probs.append(model.get_probs(new_x))
            return tf.reduce_mean(tf.stack(probs, 0), 0)

    elif method == 'logits':
        return model.get_logits(x)


def do_sess_batched_eval(sess, x, x_gen, X_in, X_out_shape, args=None):
    """
    Apply gen to ndarray, and return a new ndarray.
    :param sess: TF session to use
    :param x_gen: generator feed with placeholder, eg. attack(x)
    :param X_in: numpy array with inputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
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
            feed_dict = {x: X_in[start:end]}
            X_out[start:end] = x_gen.eval(feed_dict=feed_dict)

        assert end >= len(X_in)

    return X_out


def get_merged_train_data(sess, x, attack_dict, X_train, eval_params, attack_types):
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
    Y_merged = np.zeros((X_train.shape[0], len(attack_types) + 1))
    Y_merged[:, 0] = np.ones(X_train.shape[0])
    for i, attack_name in enumerate(attack_types):
        # generate adversrial X
        X_train_adv = do_sess_batched_eval(
            sess, x, attack_dict[attack_name](x), X_train, X_train.shape, args=eval_params)
        # constructe Y
        Y_train_adv = np.zeros((X_train.shape[0], len(attack_types) + 1))
        Y_train_adv[:, i+1] = np.ones(X_train.shape[0])
        # merge them to X_merged and Y_merged
        X_merged = np.vstack((X_merged, X_train_adv))
        Y_merged = np.vstack((Y_merged, Y_train_adv))

    return X_merged, Y_merged

def test_detector(sess, x, detector, x_in, X_test, X_out_shape, args, fp=None, random=False, k=5, stddev=1.0):
    det_probs = get_pred(x_in, detector, 'probs',
                         random=random, k=k, stddev=stddev)
    Probs = do_sess_batched_eval(
        sess, x, det_probs, X_test, X_out_shape, args=args)

    Preds = np.argmax(Probs, axis=1)
    unique, counts = np.unique(Preds, return_counts=True)

    if fp is not None:
        fp.write('detector mean probs:' + str(np.mean(Probs, axis=0)) + '\n')
        fp.write('detector preds counts:' +
                 str(dict(zip(unique, counts))) + '\n')
    else:
        print('detector mean probs:' + str(np.mean(Probs, axis=0)) + '\n')
        print('detector preds counts:' + str(dict(zip(unique, counts))))


def train_detector(sess, x, name, attack_dict, X_train, eval_params, attack_types, dataset, nb_filters, input_shape, label_smoothing, train_params, rng, eval_detector):
    # Make and train detector that classfies whcih source (clean or adv_1 or adv_2...)
    # prepare data
    print('Preparing merged data...')
    X_merged, Y_merged = get_merged_train_data(
        sess, x, attack_dict, X_train, eval_params, attack_types)
    # train the detector
    if dataset == 'mnist':
        detector = ModelBasicCNN(
            'detector_'+name, (len(attack_types) + 1), nb_filters)
    elif dataset == 'cifar10':
        detector = ModelAllConvolutional(
            'detector_'+name, (len(attack_types) + 1), nb_filters, input_shape=input_shape)

    loss_d = CrossEntropy(detector, smoothing=label_smoothing)
    print('Train detector with X_merged and Y_merged shape:',
          X_merged.shape, Y_merged.shape)

    # eval detector
    if eval_detector:
        X_merged_train, X_merged_test, Y_merged_train, Y_merged_test = train_test_split(
            X_merged, Y_merged, test_size=0.20, random_state=42)
        train(sess, loss_d, X_merged_train, Y_merged_train,
              args=train_params, rng=rng, var_list=detector.get_params())
        do_eval(sess, x, tf.placeholder(tf.float32, shape=(None, (len(attack_types) + 1))),
                do_preds(x, detector, 'probs'), X_merged_test, Y_merged_test, 'Detector accuracy', eval_params)
    else:
        train(sess, loss_d, X_merged, Y_merged,
              args=train_params, rng=rng, var_list=detector.get_params())

    return detector


def write_exp_summary(fp, flags, is_online, pred_random, k, stddev, reinforcement_ens):
    string = 'Experiment for AML project.\n' + \
             'Date: ' + str(datetime.datetime.now()) + '\n\n' + \
             'Dataset: ' + flags.dataset + '\n' + \
             'Attack types: ' + str(flags.attack_type) + '\n' + \
             'Model type: ' + str(flags.attack_model) + '\n' + \
             'Is Online: ' + str(is_online) + '\n' + \
             'Randomness in prediction: ' + str(pred_random) + '\n' + \
             'Random K: ' + str(k) + '\n' + \
             'Stddev: ' + str(stddev) + '\n' + \
             'Ensemble reinforcement: ' + str(reinforcement_ens) + '\n' + \
             '\n\n'
    fp.write(string)
