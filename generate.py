
import numpy as np
import logging
import math
import itertools
from tqdm import tqdm


import tensorflow as tf
from cleverhans.attacks import *
from cleverhans.utils_mnist import data_mnist
from cleverhans.dataset import CIFAR10
#from utils_fmnist import data_fmnist
from models.all_cnn import ModelAllConvolutional
from models.basic_model import ModelBasicCNN
from models.cifar10_model import make_wresnet
from models.mnist_model import MadryMNIST


def get_model(dataset, attack_model, scope, nb_classes, nb_filters, input_shape):
    print(dataset, attack_model,nb_classes)
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

# function to get the adv_x
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

# function to get the parameters for adv_x
def get_para(dataset, attack_type, att=False):
    if dataset == 'mnist' or 'fmnist':
        print('attack_type', attack_type)
        attack_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}
        if attack_type == 'fgsm':
            attack_params = attack_params
        elif attack_type == 'bim':
            attack_params.update({'nb_iter': 50, 'eps_iter': .01})
        elif attack_type == 'pgd':
            if att == False:
                attack_params.update({'eps_iter': 0.05, 'nb_iter': 10})
            else:
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
                if att == False:
                    attack_params.update({'eps_iter': 2, 'nb_iter': 10})
                else:
                    attack_params.update({'eps_iter': 2, 'nb_iter': 20})
            elif attack_type == 'jsma':
                attack_params.update({'theta': 1., 'gamma': 0.1,
                                      'clip_min': 0., 'clip_max': 1.,
                                      'y_target': None})
            else:
                raise ValueError(attack_type)
    return attack_params
