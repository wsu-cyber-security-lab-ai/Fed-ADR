# import mxnet as mx
# import numpy as np
from copy import deepcopy
import time
# from numpy import random
# from mxnet import nd, autograd, gluon

import numpy as np

# def no_byz(v, f):
#     return v

def partial_trim(v, f):
    '''
    Partial-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised. 
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    # first compute the statistics
    vi_shape = v[0].shape
    all_grads = np.concatenate(v, axis=1)
    adv_grads = all_grads[:, :f]
    e_mu = np.mean(adv_grads, axis=1)  # mean
    e_sigma = np.sqrt(np.sum(np.square(adv_grads - e_mu.reshape(-1, 1)), axis=1) / f)  # standard deviation

    # for i in range(f):
    #     # apply attack to compromised worker devices with randomness
    #     v[i] = (e_mu - np.multiply(e_sigma, np.sign(e_mu)) * 3.5).reshape(vi_shape)

    return e_mu, e_sigma, vi_shape

# def full_trim(v, f):
#     '''
#     Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
#     v: the list of squeezed gradients
#     f: the number of compromised worker devices
#     '''
#     # first compute the statistics
#     vi_shape = v[0].shape
#     v_tran = nd.concat(*v, dim=1)
#     maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
#     minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
#     direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
#     directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

#     for i in range(f):
#         # apply attack to compromised worker devices with randomness
#         random_12 = 2
#         v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
#     return v


# def gaussian_attack(v, f, epsilon=0.01):
#     vi_shape = v[0].shape
#     adv_grads = nd.concat(*v, dim=1)
#     e_mu = nd.mean(adv_grads, axis=1)
#     e_sigma = nd.sqrt(nd.sum(nd.square(nd.subtract(adv_grads, e_mu.reshape(-1, 1))), axis=1) / f)
#     for i in range(f):
#         norm = nd.norm(v[i])
#         v[i] = nd.random.normal(e_mu, e_sigma).reshape(vi_shape)
#         v[i] *= norm/nd.norm(v[i])
#     return v


# def mean_attack(v, f, epsilon=0.01):

#     for i in range(f):
#         v[i] = -v[i]
#     return v


# def full_mean_attack(v, f, epsilon=0.01):
#     if f == len(v):
#         for i in range(f):
#             v[i] = -v[i]
#         return v

#     if f == 0:
#         return v

#     vi_shape = v[0].shape
#     # first compute the distribution parameters
#     all_grads = nd.concat(*v, dim=1)
#     grad_sum = nd.sum(all_grads, axis=1)
#     benign_grad_sum = nd.sum(all_grads[:, f:], axis=1)
#     for i in range(f):
#         v[i] = ((-grad_sum - benign_grad_sum) / f).reshape(vi_shape)
#     return v


def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)
    sorted_distance = np.sort(np.sum(np.square(v - gradient), axis=0))
    return np.sum(sorted_distance[1:(1+num_neighbours)])


def krum(v, f):
    if len(v) - f - 2 <= 0:
        f = len(v) - 3
    if len(v[0].shape) > 1:
        v_tran = np.concatenate(v, axis=1)
    else:
        v_tran = v
    scores = np.array([score(gradient, v_tran, f) for gradient in v])
    min_idx = int(np.argmin(scores, axis=0))
    krum_nd = np.reshape(v[min_idx], (-1,))
    return min_idx, krum_nd


# f is the total number of malicoius node
# v is the values for all nodes (malicoius + benign), first f is the malicoius
def dir_partial_krum_lambda(v, f, epsilon=0.01):
    vi_shape = v[0].shape

    v_tran = np.transpose(np.concatenate(v, axis=1))[:f].copy()
    original_dir = np.mean(v_tran, axis=0).reshape(vi_shape)
    v_attack_number = 1

    lamda = 1.0

    while v_attack_number < f:
        lamda = 1.0
        v_simulation = [each_v.copy() for each_v in v[:f]]

        for i in range(v_attack_number):
            v_simulation.append(-lamda * np.sign(original_dir))

        min_idx, _ = krum(v_simulation, v_attack_number)
        stop_threshold = 0.00002
        while min_idx < f and lamda > stop_threshold:
            lamda /= 2
            for i in range(f, f + v_attack_number):
                v_simulation[i] = -lamda * np.sign(original_dir)
            min_idx, _ = krum(v_simulation, v_attack_number)

        v_attack_number += 1

        if min_idx >= f:
            break

    # print('chosen lambda:', lamda)
    
    # v[0] = -lamda * np.sign(original_dir)
    
    # for i in range(1, f):
    #     random_raw = np.random.uniform(size=vi_shape) - 0.5
    #     random_norm = np.random.uniform() * epsilon
    #     randomness = (random_raw * random_norm) / np.linalg.norm(random_raw)
    #     v[i] = -lamda * np.sign(original_dir) + randomness

    return lamda, original_dir, vi_shape


# def dir_full_krum_lambda(v, f, epsilon=0.01):
#     # when there are fewer than 2 clients, a random local model update is selected
#     # therefore, the strongest attack is to flip the sign of the local model update
#     if len(v) <= 2:
#         for i in range(f):
#             v[i] = -v[i]
#         return v

#     vi_shape = v[0].shape
#     v_tran = nd.transpose(nd.concat(*v, dim=1)).copy()
#     # v_tran = nd.transpose(nd.concat(*v, dim=1))[:f].copy()
#     # original_dir = nd.mean(v_tran, axis=0).reshape(vi_shape)
#     _, original_dir = krum(v, f)
#     original_dir = original_dir.reshape(vi_shape)

#     lamda = 1.
#     for i in range(f):
#         v[i] = -lamda * nd.sign(original_dir)
#     min_idx, _ = krum(v, f)
#     stop_threshold = 1e-5
#     while (min_idx >= f and lamda > stop_threshold):
#         lamda = lamda / 2
#         for i in range(f):
#             v[i] = -lamda * nd.sign(original_dir)
#         min_idx, _ = krum(v, f)

#     print('chosen lambda:', lamda)
#     v[0] = -lamda * nd.sign(original_dir)
#     for i in range(1, f):
#         random_raw = nd.random.uniform(shape=vi_shape) - 0.5
#         random_norm = nd.random.uniform().asscalar() * epsilon
#         randomness = random_raw * random_norm / nd.norm(random_raw)
#         v[i] = -lamda * nd.sign(original_dir) + randomness

#     return v

# def scaling_attack(v, f, epsilon=0.01):
#     scaling_factor = len(v)
#     for param_id in range(f):
#         v[param_id] = v[param_id]*scaling_factor
#     return v