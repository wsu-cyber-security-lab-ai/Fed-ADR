# import mxnet as mx
# import numpy as np
from copy import deepcopy
import time
# from numpy import random
# from mxnet import nd, autograd, gluon

import numpy as np
from utils import utils
import torch

import cupy as cp

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
    
    # Compute the mean with keepdims=True to preserve the (n,1) shape.
    e_mu = np.mean(adv_grads, axis=1, keepdims=True)
    
    # Compute the standard deviation similarly, subtracting e_mu and keeping dims.
    e_sigma = np.sqrt(np.sum(np.square(adv_grads - e_mu), axis=1, keepdims=True) / f)
    
    return e_mu, e_sigma, vi_shape

def attack_trim(v, f):
    vi_shape = v[0].shape

    # Convert list of NumPy arrays into a single array with columns stacked
    v_tran = np.stack(v, axis=1)  # Shape: (num_features, num_workers)
    
    # Compute max and min along the second axis (workers dimension)
    maximum_dim = np.max(v_tran, axis=1, keepdims=True)
    minimum_dim = np.min(v_tran, axis=1, keepdims=True)
    
    # Reshape maximum_dim and minimum_dim to match vi_shape
    maximum_dim = maximum_dim.reshape(vi_shape)
    minimum_dim = minimum_dim.reshape(vi_shape)

    # Compute the direction of the summed gradients
    direction = np.sign(np.sum(v_tran, axis=1, keepdims=True))  # Shape: (num_features, 1)
    direction = direction.reshape(vi_shape)  # Reshape to match vi_shape

    # Determine the attack direction
    directed_dim = np.where(direction > 0, minimum_dim, maximum_dim)  # Shape: vi_shape
    
    return directed_dim, direction, vi_shape

# def gaussian_attack(v, f, scale_factor=5.0):
#     vi_shape = v[0].shape
#     adv_grads = np.concatenate(v, axis=1)
    
#     e_mu = np.mean(adv_grads, axis=1, keepdims=True)
#     # Increase attack strength with scale_factor
#     e_sigma = scale_factor * np.std(adv_grads, axis=1, keepdims=True)

#     return e_mu, e_sigma, vi_shape

def gaussian_attack(v, f, scale_factor=5.0):
    # Convert NumPy arrays to CuPy arrays
    v = [cp.asarray(x) for x in v]

    vi_shape = v[0].shape
    adv_grads = cp.concatenate(v, axis=1)

    e_mu = cp.mean(adv_grads, axis=1, keepdims=True)
    e_sigma = scale_factor * cp.std(adv_grads, axis=1, keepdims=True)

    return e_mu, e_sigma, vi_shape

def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)
    sorted_distance = np.sort(np.sum(np.square(v - gradient), axis=0))
    return np.sum(sorted_distance[1:(1+num_neighbours)])


# def krum(v, f):
#     # if len(v) - f - 2 <= 0:
#     #     f = len(v) - 3
#     if len(v[0].shape) > 1:
#         v_tran = np.concatenate(v, axis=1)
#     else:
#         v_tran = v
#     scores = np.array([score(gradient, v_tran, f) for gradient in v])
#     min_idx = int(np.argmin(scores, axis=0))
#     krum_nd = np.reshape(v[min_idx], (-1,))
#     return min_idx, krum_nd

def krum(v, f):
    start_time = time.time()
    # Move everything to GPU
    # v = [cp.asarray(x) for x in v]

    if len(v[0].shape) > 1:
        v_tran = cp.concatenate(v, axis=1)
    else:
        v_tran = cp.stack(v)  # stack for 1D arrays
    
    scores = cp.array([score(gradient, v_tran, f) for gradient in v])

    min_idx = int(cp.argmin(scores, axis=0).get())  # .get() to bring result back to CPU
    krum_nd = cp.reshape(v[min_idx], (-1,))

    end_time = time.time()  # End timer
    print(f"Time spent in my_function: {end_time - start_time:.4f} seconds")

    return min_idx, krum_nd

import cupy as cp
import time

def fast_krum_memory_efficient(v, f):
    if len(v[0].shape) > 1:
        v_tran = cp.concatenate(v, axis=1).T
    else:
        v_tran = cp.stack(v)

    num_grads = v_tran.shape[0]

    # 1. Precompute squared norms
    norms = cp.sum(v_tran ** 2, axis=1)  # shape (num_grads,)

    # 2. Compute pairwise distance matrix using (a-b)^2 = ||a||^2 + ||b||^2 - 2<a,b>
    dists = norms[:, None] + norms[None, :] - 2 * (v_tran @ v_tran.T)  # (num_grads, num_grads)

    # Numerical stability fix: clip small negatives to zero
    dists = cp.maximum(dists, 0)

    # 3. For each gradient, sort distances to others
    sorted_distances = cp.sort(dists, axis=1)

    # 4. Sum the closest (n - f - 2) distances
    num_neigh = num_grads - f - 2
    scores = cp.sum(sorted_distances[:, 1:1+num_neigh], axis=1)  # skip self-distance at index 0

    min_idx = int(cp.argmin(scores))
    krum_nd = cp.reshape(v[min_idx], (-1,))

    return min_idx, krum_nd


def krum2(clients, f):

    input = torch.tensor(np.array(clients))
    input = input.permute(2, 1, 0)

    n = input.shape[-1]
    # f = n // 2  # worse case 50% malicious points
    k = n - f - 2

    # collection distance, distance from points to points
    x = input.permute(0, 2, 1)
    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    # krum
    krum = input[:, :, [i_star]]
    # Multi-Krum
    mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)

    return i_star, krum

    # (11, 81194, 1)

# def dir_partial_krum_lambda(v, f):
#     start_time = time.time()
#     print(start_time)
#     vi_shape = v[0].shape

#     v_tran = np.transpose(np.concatenate(v, axis=1))[:f].copy()
#     original_dir = np.mean(v_tran, axis=0).reshape(vi_shape)
#     v_attack_number = 1

#     lamda = 5.0

#     while v_attack_number < f:
#         lamda = 5.0
#         v_simulation = [each_v.copy() for each_v in v[:f]]

#         for i in range(v_attack_number):
#             v_simulation.append(-lamda * np.sign(original_dir))

#         min_idx, _ = krum(v_simulation, v_attack_number)
#         stop_threshold = 0.00002
#         while min_idx < f and lamda > stop_threshold:
#             lamda /= 2
#             for i in range(f, f + v_attack_number):
#                 v_simulation[i] = -lamda * np.sign(original_dir)
#             min_idx, _ = krum(v_simulation, v_attack_number)

#         v_attack_number += 1

#         if min_idx >= f:
#             break

#     end_time = time.time()  # End timer
#     print(f"Time spent in my_function: {end_time - start_time:.4f} seconds")
#     return lamda, original_dir, vi_shape

import cupy as cp

def dir_partial_krum_lambda(v, f):
    start_time = time.time()
    vi_shape = v[0].shape
    v = [cp.asarray(x) for x in v]  # Convert each numpy array to cupy
    v_concat = cp.concatenate(v, axis=1)
    v_tran = cp.transpose(v_concat)[:f].copy()
    original_dir = cp.mean(v_tran, axis=0).reshape(vi_shape)
    v_attack_number = 1

    # lamda = 5.0 # mnist and fashion mnist
    lamda = 50.0 # cifar

    while v_attack_number < f:
        # lamda = 5.0 # mnist and fashion mnist
        lamda = 50.0 # cifar
        v_simulation = [each_v.copy() for each_v in v[:f]]

        for _ in range(v_attack_number):
            v_simulation.append(-lamda * cp.sign(original_dir))

        min_idx, _ = fast_krum_memory_efficient(v_simulation, v_attack_number)
        stop_threshold = 0.00002

        while min_idx < f and lamda > stop_threshold:
            lamda /= 2
            attack_vector = -lamda * cp.sign(original_dir)

            for i in range(f, f + v_attack_number):
                v_simulation[i] = attack_vector

            min_idx, _ = fast_krum_memory_efficient(v_simulation, v_attack_number)

        v_attack_number += 1

        if min_idx >= f:
            break
    end_time = time.time()  # End timer
    print(f"Time spent in my_function: {end_time - start_time:.4f} seconds")
    return lamda, original_dir, vi_shape

def krum_attack(v, lr, f):
    if f == 0:
        return v
    else:
        vi_shape = v[0].shape

        _, original_dir = krum(v, f)
        original_dir = original_dir.reshape(vi_shape)

        lamda = 0.25
        for i in range(f):
            v[i] = -lamda * np.sign(original_dir)
        min_idx, _ = krum(v, f)
        stop_threshold = 0.00001 * 2 / lr
        while (min_idx >= f and lamda > stop_threshold):
            lamda = lamda / 2
            for i in range(f):
                v[i] = -lamda * np.sign(original_dir)
            min_idx, _ = krum(v, f)
    return lamda, original_dir, vi_shape
