import tensorflow as tf
from cleverhans.utils_tf import model_eval
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
import numpy as np
import math

def do_eval(sess, x, y, pred, X_test, Y_test, message, eval_params):
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
    print(message)
    acc = model_eval(
        sess, x, y, pred, X_test, Y_test, args=eval_params)
    print('Accuracy: %0.4f\n' % acc)


def do_probs(x, model, def_model_list=None):
    """
    Return the mean prob of multiple models
    """
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    probs = [m.get_probs(x) for m in models]
    return tf.reduce_mean(probs, 0)


def do_logits(x, model, def_model_list=None):
    """
    Return the mean logits of multiple models
    """
    if def_model_list is not None:
        models = [model] + def_model_list
    else:
        models = [model]
    logits = [m.get_logits(x) for m in models]
    return tf.reduce_mean(logits, 0)


def do_transform(sess, x, x_gen, X_in, feed=None, args=None):
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
        X_out = np.zeros_like(X_in)
        print('X in shape:', X_in.shape)
        print('X out shape:', X_out.shape)
        for batch in range(nb_batches):
            start = batch * args.batch_size
            end = min(len(X_in), start + args.batch_size)
            feed_dict = {x: X_in[start:end]}
            if feed is not None:
                feed_dict.update(feed)
            X_out[start:end] = x_gen.eval(feed_dict=feed_dict)

        assert end >= len(X_in)

    return X_out

            