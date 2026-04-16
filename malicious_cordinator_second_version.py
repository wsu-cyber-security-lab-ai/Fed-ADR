from __future__ import print_function

import random
import statistics
from numpy.linalg import norm
from collections import OrderedDict,Counter,defaultdict
from sklearn.decomposition import TruncatedSVD
import torch
from copy import deepcopy
from utils import utils

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans, DBSCAN
import numpy as np

import time
from attack_types import AttackTypes
from aggregation_types import AggregationTypes
import byzantine
from sklearn.impute import SimpleImputer

import math

import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN
from collections import Counter

class MaliciousCordinatorSecondVersion():
    def __init__(self):
        self.attackerNodeBeforeAttackGradient = {}
        self.attackerNodeBeforeAttackWeight = {}
        self.attackerNodeAttackerType = {}
        self.attackerNodeAvgDelta = {}
        self.attackerNodeSumDelta = {}
        self.attackerNodehogAvgLen = {}

        self.attackerNodeOfLabelFlipping = []
        self.attackerNodeOfBackdoor = []
        self.attackerNodeOfMultiLabelFlipping = []

        self.normalNodeDummyValue = {}
        self.normalNodeDummyAvgDelta = {}
        self.normalNodeDummySumDelta = {}
        self.normalNodeDummyhogAvgLen = {}

        self.attackerNodes = set()
        self.numberOfClientsTraining = 0
        self.numberOfAttackerNodesAccessable = 0
        self.numberOfAttackerNodesInCurrentIteration = 0
        self.model = None

        self.minimumNumberOfMaliciousClientsThreshold = 2
        self.minimumIterationToStartTheAttack = 0
        self.lastModelAccuracyThreshold = 40
        self.samplesThreshold = 5

        self.aggregationMethod = None

        self.attackerNodeChoosenGradientInThisIteration = {}
        self.attackerNodeChoosenWeightInThisIteration = {}
        self.attackerNodeChoosenAvgDeltaInThisIteration = {}
        self.attackerNodeChoosenSumDeltaInThisIteration = {}
        self.attackerNodeChoosenhogAvgLenInThisIteration = {}
        self.currentIteration = 0

        # MNIST uses default eps=0.5, min_sample=5
        # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5
        self.delay_decision = 2

        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()

        self.lastModelAccuracy = 0

        self.emptyStates = None

        self.krumLamda = None
        self.original_dir = None

        self.vi_shape = None

        self.e_sigma = None
        self.e_mu = None
        self.last_f = None

        self.directed_dim = None
        self.direction = None

    def reset(self):
        self.attackerNodes = set()
        self.attackerNodeBeforeAttackGradient = {}
        self.attackerNodeBeforeAttackWeight = {}
        self.attackerNodeAttackerType = {}
        self.attackerNodeAvgDelta = {}
        self.attackerNodeSumDelta = {}
        self.attackerNodehogAvgLen = {}

        self.attackerNodeOfLabelFlipping = []
        self.attackerNodeOfBackdoor = []
        self.attackerNodeOfMultiLabelFlipping = []

        self.normalNodeDummyValue = {}
        self.normalNodeDummyAvgDelta = {}
        self.normalNodeDummySumDelta = {}
        self.normalNodeDummyhogAvgLen = {}
    

    def addMaliciousClientUpdate(self, client, attackerType):
        self.attackerNodes.append(client.cid)
        self.numberOfAttackerNodesInCurrentIteration += 1

        trainable_parameter = utils.getTrainableParameters(self.model)

        # client.model.cpu()
        clientState = client.model.state_dict()
        clientAvgDelta = client.avg_delta
        clientSumDelta = client.sum_hog
        clientStateChange = deepcopy(self.emptyStates)

        self.attackerNodeBeforeAttackGradient[client.cid] = {}
        self.attackerNodeBeforeAttackWeight[client.cid] = {}
        self.attackerNodeAvgDelta[client.cid] = {}
        self.attackerNodeSumDelta[client.cid] = {}
        

        for p in client.originalState:
            if p not in trainable_parameter:
                continue
            # clientStateChange[p] = clientState[p] - client.originalState[p]
            clientStateChange[p] = clientState[p] - client.originalState[p].to(clientState[p].device)
            # clientStateChange[p] = clientState[p].cpu() - client.originalState[p].cpu()
            self.attackerNodeBeforeAttackGradient[client.cid][p] = deepcopy(clientStateChange[p])
            self.attackerNodeBeforeAttackWeight[client.cid][p] = deepcopy(clientState[p])
            self.attackerNodeAvgDelta[client.cid][p] = deepcopy(clientAvgDelta[p])
            self.attackerNodeSumDelta[client.cid][p] = deepcopy(clientSumDelta[p])

        # print(self.attackerNodeBeforeAttackWeight[client.cid].keys())

        # self.attackerNodeBeforeAttackGradient[client.cid] = deepcopy(clientStateChange)
        # self.attackerNodeBeforeAttackWeight[client.cid] = deepcopy(clientState)
        # self.attackerNodeAvgDelta[client.cid] = deepcopy(clientAvgDelta)
        # self.attackerNodeSumDelta[client.cid] = deepcopy(clientSumDelta)
        self.attackerNodehogAvgLen[client.cid] = len(client.hog_avg)
        self.attackerNodeAttackerType[client.cid] = attackerType

        # print(attackerType)

        if attackerType == AttackTypes.LabelFlipping:
            self.attackerNodeOfLabelFlipping.append(client.cid)

        if attackerType == AttackTypes.BackDoor:
            self.attackerNodeOfBackdoor.append(client.cid)

        if attackerType == AttackTypes.MultiLabelFlipping:
            self.attackerNodeOfMultiLabelFlipping.append(client.cid)

    def calculateNewMalicoiusAttack(self, client, attackerType):
        if self.aggregationMethod == AggregationTypes.MudHog:
            result = self.formulate_mudhog_attack(client, attackerType)
            return result
        elif self.aggregationMethod == AggregationTypes.Krum:
            result = self.formulate_krum_attack(client, attackerType)
            return result
        elif self.aggregationMethod == AggregationTypes.Trimmed:
            result = self.formulate_trim_attack(client, attackerType)
            return result
    
    def formulate_mudhog_attack(self, client, attackType):
        result = {
            "newMaliciousClient": None,
            "isNewMalicoiusClientCalculated": False,
        }

        # if self.currentIteration == 22:
        #     print(client.model.state_dict())

        if  (
            len(self.attackerNodeBeforeAttackGradient) == self.numberOfAttackerNodesInCurrentIteration and
            self.currentIteration>=self.minimumIterationToStartTheAttack and 
            self.numberOfAttackerNodesInCurrentIteration>=self.minimumNumberOfMaliciousClientsThreshold
            ):
        # if  self.lastModelAccuracy>=self.lastModelAccuracyThreshold and len(self.attackerNodeBeforeAttackGradient) == self.numberOfAttackerNodesInCurrentIteration and self.currentIteration>=self.minimumIterationToStartTheAttack and self.numberOfAttackerNodesInCurrentIteration>=self.minimumNumberOfMaliciousClientsThreshold:
        # if  self.currentIteration % 3 != 0 and len(self.attackerNodeBeforeAttackGradient) == self.numberOfAttackerNodesInCurrentIteration and self.currentIteration>=self.minimumIterationToStartTheAttack and self.numberOfAttackerNodesInCurrentIteration>=self.minimumNumberOfMaliciousClientsThreshold:
            dict1 = dict(self.attackerNodeChoosenGradientInThisIteration)
            dict2 = self.attackerNodeBeforeAttackGradient
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenWeightInThisIteration)
            dict2 = self.attackerNodeBeforeAttackWeight
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenAvgDeltaInThisIteration)
            dict2 = self.attackerNodeAvgDelta
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenSumDeltaInThisIteration)
            dict2 = self.attackerNodeSumDelta
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenhogAvgLenInThisIteration)
            dict2 = self.attackerNodehogAvgLen
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            self.normalNodeDummyValue = {}
            self.normalNodeDummyAvgDelta = {}
            self.normalNodeDummySumDelta = {}
            self.normalNodeDummyhogAvgLen = {}

            listOfMaliciousClientsIdsToChooseFrom = list(self.attackerNodeBeforeAttackGradient.keys())

            trainable_parameter = utils.getTrainableParameters(self.model)

            # avgDect1 = self.normalNodeDummyValue[k]
            #     avgDect2 = self.normalNodeDummyAvgDelta[k]

            for i in range(self.numberOfClientsTraining):
                if i not in self.attackerNodes:
                    selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
                    self.normalNodeDummyValue[i] = {}
                    self.normalNodeDummyAvgDelta[i] = {}
                    self.normalNodeDummySumDelta[i] = {}

                    for p in trainable_parameter:
                        # print(p)
                        # print(self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey].keys())
                        # print(len(self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey][p]))
                        # print(self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey][p])
                        # if p in self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey]:
                        self.normalNodeDummyValue[i][p] = deepcopy(
                            self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey][p]
                        )
                        self.normalNodeDummyAvgDelta[i][p] = deepcopy(
                            self.attackerNodeAvgDelta[selectedRandomMaliciousKey][p]
                        )
                        self.normalNodeDummySumDelta[i][p] = deepcopy(
                            self.attackerNodeSumDelta[selectedRandomMaliciousKey][p]
                        )
                    # self.normalNodeDummyValue[i] = deepcopy(self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey])
                    # self.normalNodeDummyAvgDelta[i] = deepcopy(self.attackerNodeAvgDelta[selectedRandomMaliciousKey])
                    # self.normalNodeDummySumDelta[i] = deepcopy(self.attackerNodeSumDelta[selectedRandomMaliciousKey])
                    self.normalNodeDummyhogAvgLen[i] = deepcopy(self.attackerNodehogAvgLen[selectedRandomMaliciousKey])

            # TODO stable point 1

            # STAGE 1: Collect long and short HoGs.
            main_short_HoGs = {}
            main_long_HoGs = {}
            main_normalized_sHoGs = {}
            main_full_norm_short_HoGs = []
            keys = self.attackerNodeBeforeAttackGradient.keys()
            attackerKey = list(keys)[self.numberOfAttackerNodesInCurrentIteration-1]

            main_short_HoGs, main_long_HoGs, main_normalized_sHoGs, main_full_norm_short_HoGs = self.collect_short_long_hogs()


            # for key, value in main_short_HoGs.items():
            #     try:
            #         print(f"{key}: shape = {value.shape}")
            #     except AttributeError:
            #         print(f"{key}: type = {type(value)}")

            # STAGE 2: Formalate the attack
            newMaliciousClient = {}
            scoreOfNodesMarkedAsMalicious = 0
            newCoordinatorPossibleAttacks = {}

            listOfNewMaliciousClient = {}
            listOfNumberOfClientMarkedAsMalicious = {}
            listOfClientsMarkedAsMalicious = {}
            listOfNewCoordinatorPossibleAttacks = {}
            
            # print(trainable_parameter)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # if attackerType == AttackTypes.GuassianAttack:
            if self.aggregationMethod == AggregationTypes.MudHog:
                numLow = {}
                numHigh = {}

                numLow, numHigh = self.get_guassian_low_and_high_weight()

                # for t in range(self.samplesThreshold):
                for t in range(1):
                    newMaliciousClientWeight = {}

                    # for p in self.model.state_dict():
                    #     if p not in trainable_parameter:
                    #         continue
                    #     # Move arrays to GPU
                    #     numLow_gpu = cp.asarray(numLow[p])
                    #     numHigh_gpu = cp.asarray(numHigh[p])

                    #     # Compute step size for each element
                    #     difference = numHigh_gpu - numLow_gpu
                    #     step = difference / self.samplesThreshold

                    #     # Midpoint index
                    #     mid_index = self.samplesThreshold // 2

                    #     # Vectorized tempRange equivalent using broadcasting
                    #     newSelectedSampledValues = numLow_gpu + step * mid_index

                    #     # Flip sign (always -1 as per your logic)
                    #     newSelectedSampledValues = newSelectedSampledValues * -1

                    #     # Handle edge cases where step is 0 or NaN (retain original value)
                    #     mask_invalid = (step == 0) | cp.isnan(step)
                    #     newSelectedSampledValues[mask_invalid] = numLow_gpu[mask_invalid]

                    #     # Convert to torch tensor on GPU
                    #     newMaliciousClientWeight[p] = torch.tensor(cp.asnumpy(newSelectedSampledValues), device=device)


                    # for p in self.model.state_dict():
                    #     if p not in trainable_parameter:
                    #         continue
                    #     newSelectedSampledValues = cp.zeros_like(numLow[p])

                    #     # Get all indices at once
                    #     all_indices = cp.array(cp.nonzero(cp.ones_like(numLow[p]))).T  # shape (N, ndim)

                    #     for idx in all_indices:
                    #         idx = tuple(idx.tolist())
                    #         rangeStart = numLow[p][idx]
                    #         rangeEnd = numHigh[p][idx]

                    #         difference = rangeEnd - rangeStart
                    #         step = difference / self.samplesThreshold

                    #         if step != 0 and (not cp.isnan(step)):
                    #             tempRange = cp.arange(rangeStart, rangeEnd, step)
                    #             sampledRangeOfValues = tempRange
                    #             flip = cp.random.choice(cp.array([-1, -1]), size=1)[0]
                    #             new_value = sampledRangeOfValues[self.samplesThreshold // 2] * flip
                    #             newSelectedSampledValues[idx] = new_value
                    #         else:
                    #             newSelectedSampledValues[idx] = rangeStart

                    #     newMaliciousClientWeight[p] = torch.tensor(cp.asnumpy(newSelectedSampledValues), device='cuda')



                    for p in self.model.state_dict():
                        if p not in trainable_parameter:
                            continue
                        newSelectedSampledValues = np.full_like(numLow[p], 0)

                        for index, value in np.ndenumerate(numLow[p]):
                            rangeStart = value
                            rangeEnd = numHigh[p][index]

                            difference = rangeEnd - rangeStart
                            step = difference / self.samplesThreshold

                            if step !=0 and (not np.isnan(step)):
                            # if step !=0:
                                # print(rangeStart)
                                # print(rangeEnd)
                                # print(step)
                                # print()
                                tempRange = np.arange(rangeStart, rangeEnd, step)
                                sampledRangeOfValues = list(tempRange)
                                flip = np.random.choice([-1, -1])
                                # random.shuffle(sampledRangeOfValues)
                                # new_value = sampledRangeOfValues[t] * flip
                                new_value = sampledRangeOfValues[self.samplesThreshold // 2] * flip
                                # new_value = value * flip
                                # new_value = listOfParamters[p][len(self.attackerNodeBeforeAttackGradient)-1][index] * flip
                                newSelectedSampledValues[index] = new_value
                            else:
                                newSelectedSampledValues[index] = value

                        newMaliciousClientWeight[p] = torch.tensor(newSelectedSampledValues)

                    coordinatorNewAttacksLodge = {
                        "pre_mal_id": deepcopy(self.pre_mal_id),
                        "mal_ids": deepcopy(self.mal_ids),
                        "count_unreliable": deepcopy(self.count_unreliable),
                        "delay_decision": deepcopy(self.delay_decision),
                        "flip_sign_ids": deepcopy(self.flip_sign_ids),
                        "uAtk_ids": deepcopy(self.uAtk_ids),
                        "tAtk_ids": deepcopy(self.tAtk_ids),
                        "unreliable_ids": deepcopy(self.unreliable_ids)
                    }

                    # values1 = list(newMaliciousClientWeight.values())
                    values1 = list(newMaliciousClientWeight.values())
                    values1 = [v.cpu() for v in values1]

                    # values2 = list(dict(self.attackerNodeBeforeAttackWeight[attackerKey]).values())

                    attacker_weights = dict(self.attackerNodeBeforeAttackWeight[attackerKey])
                    values2 = [v for k, v in attacker_weights.items() if k in trainable_parameter]

                    difference = [v1 - v2 for v1, v2 in zip(values1, values2)]
                    newMaliciousClient = dict(zip(newMaliciousClientWeight.keys(), difference))

                    self.get_all_attacks(newMaliciousClient, attackerKey, coordinatorNewAttacksLodge, main_short_HoGs, main_long_HoGs, main_normalized_sHoGs, main_full_norm_short_HoGs)

                    # STAGE 2 - step 4: Calcualte best score
                    listOfNumberOfClientMarkedAsMalicious[t]  = self.calculate_attacks_score(attackerKey, coordinatorNewAttacksLodge)
                    listOfNewMaliciousClient[t] = OrderedDict(newMaliciousClientWeight)
                    listOfNewCoordinatorPossibleAttacks[t] = coordinatorNewAttacksLodge

                    # print("key", attackerKey)
                    # print("score", listOfNumberOfClientMarkedAsMalicious[t])
                    # print("t", t)
                    # print()

                # print(key)
                # print("numberOfClientMarkedAsMalicious", numberOfClientMarkedAsMalicious)
                # print()

                items = list(listOfNumberOfClientMarkedAsMalicious.items())
                larget_value_index = 0
                for i in range(len(items)):
                    larget_value_index = i
                    for j in range(i + 1, len(items)):
                        if items[j][1] >= items[larget_value_index][1]:  # Compare values
                            larget_value_index = j

                # result["newMaliciousClient"] = listOfNewMaliciousClient[larget_value_index]

                selected_malicious_client = listOfNewMaliciousClient[larget_value_index]
                for key in selected_malicious_client:
                    selected_malicious_client[key] = selected_malicious_client[key].cpu()
                result["newMaliciousClient"] = selected_malicious_client

                result["isNewMalicoiusClientCalculated"] = True

                coordinatorNewAttacksLodge = listOfNewCoordinatorPossibleAttacks[larget_value_index]

                self.pre_mal_id = coordinatorNewAttacksLodge["pre_mal_id"]
                self.mal_ids = coordinatorNewAttacksLodge["mal_ids"]
                self.count_unreliable = coordinatorNewAttacksLodge["count_unreliable"]
                self.delay_decision = coordinatorNewAttacksLodge["delay_decision"]
                self.flip_sign_ids = coordinatorNewAttacksLodge["flip_sign_ids"]
                self.uAtk_ids = coordinatorNewAttacksLodge["uAtk_ids"]
                self.tAtk_ids = coordinatorNewAttacksLodge["tAtk_ids"]
                self.unreliable_ids = coordinatorNewAttacksLodge["unreliable_ids"]

        return result

    def get_trimmed_attacks(self, grad_list, model, b=2):
        param_list = [np.concatenate([np.array(x[xx], dtype=np.float32).reshape((-1, 1)) for xx in x], axis=0) for x in grad_list]
        # Sort
        sorted_array = np.sort(np.concatenate(param_list, axis=1), axis=-1)

        # Trim
        n = len(param_list)
        m = n - 2 * b
        trim_nd = np.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=True)
        
        keys = utils.getTrainableParameters(model)
        modelData = model.state_dict()
        key_shapes = {}

        for key in keys:
            key_shapes[key] = modelData[key].shape

        Delta = deepcopy(self.emptyStates)
        DeltaMean = {}
        mean_flattened_cosine_mean = {}

        idx = 0
        for p in modelData:
            param = deepcopy(modelData[p])
            shape = modelData[p].shape
            update_value = torch.tensor(trim_nd[idx:idx + modelData[p].numel()].reshape(shape), dtype=modelData[p].dtype, device=modelData[p].device)
            param.data = update_value
            idx += modelData[p].numel()
            Delta[p] = param - modelData[p]

            arr = np.array(Delta[p])
            DeltaMean[p] = np.mean(arr, axis=0)

            A = np.array(modelData[p])
            B = np.array(DeltaMean[p])

            if len(list(A.shape))==0:
                A = np.array([modelData[p]])
                B = np.array([DeltaMean[p]])

            cosine = np.sum(A*B)/(norm(A)*norm(B))

            flattened_cosine = cosine.flatten()
            mean_flattened_cosine_mean[p] = flattened_cosine.mean()
            
        # print(type(Delta))
        # print(len(Delta))
        # print(Delta.keys())
        # print(len(DeltaMean))
        # print((DeltaMean))
        # print()
        # print(mean_flattened_cosine_mean)
        # print()

        totalCosine = 0
        for p in mean_flattened_cosine_mean:
            totalCosine+=mean_flattened_cosine_mean[p]
        
        # print(totalCosine)
        # print()

        return totalCosine

    def formulate_krum_attack(self, client, attackType):
        result = {
            "newMaliciousClient": None,
            "isNewMalicoiusClientCalculated": False,
        }

        if  (
            len(self.attackerNodeBeforeAttackGradient) == self.numberOfAttackerNodesInCurrentIteration and
            self.currentIteration>=self.minimumIterationToStartTheAttack
            # self.numberOfAttackerNodesInCurrentIteration>=self.minimumNumberOfMaliciousClientsThreshold
            ):

            dict1 = dict(self.attackerNodeChoosenGradientInThisIteration)
            dict2 = self.attackerNodeBeforeAttackGradient
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenWeightInThisIteration)
            dict2 = self.attackerNodeBeforeAttackWeight
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            self.normalNodeDummyValue = {}
            
            listOfMaliciousClientsIdsToChooseFrom = list(self.attackerNodeBeforeAttackGradient.keys())

            # for i in range(self.numberOfClientsTraining):
            #     if i not in self.attackerNodes:
            #         selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
            #         self.normalNodeDummyValue[i] = deepcopy(self.attackerNodeBeforeAttackGradient[selectedRandomMaliciousKey])

            trainable_parameter = utils.getTrainableParameters(self.model)

            for i in range(self.numberOfClientsTraining):
                if i not in self.attackerNodes:
                    selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
                    self.normalNodeDummyValue[i] = {}

                    for p in trainable_parameter:
                        self.normalNodeDummyValue[i][p] = deepcopy(
                            self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey][p]
                        )

            # TODO stable point 1

            grad_list = []

            for i in range(len(self.attackerNodes)):
                index = self.attackerNodes[i]
                temp = deepcopy(self.attackerNodeBeforeAttackGradient[index])
                grad_list.append(temp)

            for i in range(self.numberOfClientsTraining):
                if i not in self.attackerNodes:
                    temp = deepcopy(self.normalNodeDummyValue[i])
                    grad_list.append(temp)

            param_list = [np.concatenate([np.array(x[xx], dtype=np.float32).reshape((-1, 1)) for xx in x], axis=0) for x in grad_list]
            
            epsilon = 0.01
            f = self.numberOfAttackerNodesAccessable
            lr = 1e-2

            if self.krumLamda is None:
                # lamda, original_dir, vi_shape = byzantine.krum_attack(param_list, lr, f)
                lamda, original_dir, vi_shape = byzantine.dir_partial_krum_lambda(param_list, f)
                self.krumLamda = lamda
                self.original_dir = original_dir
                self.vi_shape = vi_shape
            else:
                lamda = self.krumLamda
                original_dir = self.original_dir
                vi_shape = self.vi_shape

            param_list[0] = -lamda * np.sign(original_dir)
            for i in range(1, len(self.attackerNodes)):
                # random_raw = np.random.uniform(size=vi_shape)
                # random_norm = np.random.uniform() * epsilon
                # randomness = random_raw * random_norm / np.linalg.norm(random_raw)
                random_raw = cp.random.uniform(size=vi_shape)
                random_norm = cp.random.uniform() * epsilon
                randomness = random_raw * random_norm / cp.linalg.norm(random_raw)
                # attack_strength = 1 # mnist and fashion mnist
                attack_strength = 100 # cifar
                param_list[i] = (-lamda * np.sign(original_dir)) + (randomness*attack_strength)

            keys = self.attackerNodeBeforeAttackWeight[self.attackerNodes[0]].keys()
            key_shapes = {}
            for key in keys:
                key_shapes[key] = self.attackerNodeBeforeAttackWeight[self.attackerNodes[0]][key].shape

            grad_list_recovered = []

            for p in param_list:
                grad_dict = OrderedDict()
                start_idx = 0
                
                # Split the array p based on key_shapes (multidimensional)
                for key in keys:
                    shape = key_shapes[key]
                    total_elements = torch.prod(torch.tensor(shape)).item()  # Use PyTorch's numel equivalent

                    if total_elements == (start_idx + total_elements) - start_idx:
                        # grad_dict[key] = torch.tensor(p[start_idx:start_idx + total_elements].reshape(shape), dtype=torch.float32)
                        grad_dict[key] = torch.tensor(p[start_idx:start_idx + total_elements].reshape(shape), dtype=torch.float32).cuda()
                    else:
                        grad_dict[key] = torch.zeros(shape, dtype=torch.float32)  # Fallback if dimensions mismatch
                    
                    start_idx += total_elements  # Move the start index for the next key
                
                # Append the reconstructed gradient dictionary to grad_list_recovered
                grad_list_recovered.append(grad_dict)

            # values1 = list(grad_list_recovered[len(self.attackerNodes) - 1].values())
            # values2 = list(dict(self.model.state_dict()).values())
            # difference = [v1 + v2 for v1, v2 in zip(values1, values2)]
            # newMaliciousClient = dict(zip(keys, difference))


            values1 = list(grad_list_recovered[len(self.attackerNodes) - 1].values())
            values1 = [v.cpu() for v in values1]
            attacker_weights = dict(self.model.state_dict())
            values2 = [v for k, v in attacker_weights.items() if k in trainable_parameter]
            difference = [v1 + v2 for v1, v2 in zip(values1, values2)]
            newMaliciousClient = dict(zip(keys, difference))

            result["newMaliciousClient"] = newMaliciousClient
            result["isNewMalicoiusClientCalculated"] = True

        return result
    
    def formulate_trim_attack(self, client, attackType):
        result = {
            "newMaliciousClient": None,
            "isNewMalicoiusClientCalculated": False,
        }

        if  (
            len(self.attackerNodeBeforeAttackGradient) == self.numberOfAttackerNodesInCurrentIteration and
            self.currentIteration>=self.minimumIterationToStartTheAttack
            ):

            dict1 = dict(self.attackerNodeChoosenGradientInThisIteration)
            dict2 = self.attackerNodeBeforeAttackGradient
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(self.attackerNodeChoosenWeightInThisIteration)
            dict2 = self.attackerNodeBeforeAttackWeight
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            self.normalNodeDummyValue = {}

            attackerKeys = self.attackerNodeBeforeAttackWeight.keys()
            attackerKey = list(attackerKeys)[self.numberOfAttackerNodesInCurrentIteration-1]

            listOfMaliciousClientsIdsToChooseFrom = list(self.attackerNodeBeforeAttackWeight.keys())

            trainable_parameter = utils.getTrainableParameters(self.model)

            # for i in range(self.numberOfClientsTraining):
            #     if i not in self.attackerNodes:
            #         selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
            #         self.normalNodeDummyValue[i] = deepcopy(self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey])

            for i in range(self.numberOfClientsTraining):
                if i not in self.attackerNodes:
                    selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
                    self.normalNodeDummyValue[i] = {}

                    for p in trainable_parameter:
                        self.normalNodeDummyValue[i][p] = deepcopy(
                            self.attackerNodeBeforeAttackWeight[selectedRandomMaliciousKey][p]
                        )

            grad_list = []

            for i in range(len(self.attackerNodes)):
                index = self.attackerNodes[i]
                temp = deepcopy(self.attackerNodeBeforeAttackWeight[index])
                grad_list.append(temp)

            # for i in range(self.numberOfClientsTraining):
            #     if i not in self.attackerNodes:
            #         temp = deepcopy(self.normalNodeDummyValue[i])
            #         grad_list.append(temp)

            param_list = [np.concatenate([np.array(x[xx], dtype=np.float32).reshape((-1, 1)) for xx in x], axis=0) for x in grad_list]

            f = self.numberOfAttackerNodesAccessable

            e_mu, e_sigma, vi_shape = byzantine.gaussian_attack(param_list, f, 5)

            # if self.last_f is None or self.last_f >= f:
            #     e_mu, e_sigma, vi_shape = byzantine.gaussian_attack(param_list, f)
            #     self.last_f = f
            # else:
            #     e_mu = self.e_mu
            #     e_sigma = self.e_sigma
            #     vi_shape = self.vi_shape

            # directed_dim, direction, vi_shape = byzantine.attack_trim(param_list, f)
            
            # # if  self.direction is None:
            # if  False or self.e_mu is None:
            #     # print("new")
            #     e_mu, e_sigma, vi_shape = byzantine.gaussian_attack(param_list, f)
            #     self.e_mu = e_mu
            #     self.e_sigma = e_sigma
            #     self.vi_shape = vi_shape

            #     # print(self.currentIteration)
            #     # print(e_mu)
            #     # print(e_sigma)
            #     # print(vi_shape)
            #     # print()

            #     # directed_dim, direction, vi_shape = byzantine.attack_trim(param_list, f)
            #     # self.direction = direction
            #     # self.directed_dim = directed_dim
            #     # self.vi_shape = vi_shape

            # else:
            #     # print("old")
            #     e_mu = self.e_mu
            #     e_sigma = self.e_sigma
            #     vi_shape = self.vi_shape
            #     # direction = self.direction
            #     # directed_dim = self.directed_dim

            # print(e_mu)
            # print(e_sigma)
            # print()

            for i in range(len(self.attackerNodes)):
                # Apply attack to compromised worker devices with randomness
            #     param_list[i] = (e_mu - (e_sigma * np.sign(e_mu) * 3.5)).reshape(vi_shape)

                # original = np.asarray(param_list[i])
                # norm = np.linalg.norm(original)
                # new_rand = np.random.normal(e_mu, e_sigma, size=original.shape)
                # rescaled = new_rand * (norm / np.linalg.norm(new_rand))
                # param_list[i] = rescaled

                # original = np.asarray(param_list[i])
                # attack_noise = np.random.normal(e_mu, e_sigma, size=original.shape)
                # param_list[i] = original + attack_noise * 1.0

                original = cp.asarray(param_list[i])  # Move original param to GPU
                attack_noise = cp.random.normal(e_mu, e_sigma, size=original.shape)
                param_list[i] = (original + attack_noise * 1.0).get()

                # param_list[i] = attack_noise * 1.0
                # print("noise", attack_noise)
                # print("noise", len(attack_noise))
                # print("noise", len(param_list[i]))

                # print(param_list[i]- temp)

                # param_list[i] = -attack_noise 

                # scaling_factor = 10.0  # Amplifies the attack's magnitude
                
                # # Broadcasting fix
                # expanded_mu = e_mu.reshape(-1, 1)
                # expanded_sigma = (e_sigma * scaling_factor).reshape(-1, 1)

                # # Add randomness and increase scaling
                # random_fluctuation = np.random.uniform(0.9, 1.1, size=original.shape)
                # new_rand = np.random.normal(expanded_mu, expanded_sigma * 5, size=original.shape) * random_fluctuation

                # # Remove rescaling to preserve the attack strength
                # param_list[i] = new_rand

            # for i in range(len(self.attackerNodes)):
            #     # random_12 = 2 + 1e-8
            #     random_12 = 5*np.random.uniform(size=vi_shape).astype(np.float32)
            #     param_list[i] = directed_dim * np.where(direction * directed_dim > 0, 1 / random_12, random_12)
            #     # print(vi_shape)
            #     # print(param_list[i].shape)
            #     # print(random_12.shape)


            keys = self.attackerNodeBeforeAttackWeight[self.attackerNodes[0]].keys()
            key_shapes = {}
            for key in keys:
                key_shapes[key] = self.attackerNodeBeforeAttackWeight[self.attackerNodes[0]][key].shape

            grad_list_recovered = []

            for p in param_list:
                grad_dict = OrderedDict()
                start_idx = 0
                
                # Split the array p based on key_shapes (multidimensional)
                for key in keys:
                    shape = key_shapes[key]
                    total_elements = torch.prod(torch.tensor(shape)).item()  # Use PyTorch's numel equivalent

                    if total_elements == (start_idx + total_elements) - start_idx:
                        grad_dict[key] = torch.tensor(p[start_idx:start_idx + total_elements].reshape(shape), dtype=torch.float32)
                    else:
                        grad_dict[key] = torch.zeros(shape, dtype=torch.float32)  # Fallback if dimensions mismatch
                    
                    # # Reshape the sliced part back into the original shape (torch.Tensor)
                    # grad_dict[key] = torch.tensor(p[start_idx:start_idx + total_elements].reshape(shape), dtype=torch.float32)
                    start_idx += total_elements  # Move the start index for the next key
                
                # Append the reconstructed gradient dictionary to grad_list_recovered
                grad_list_recovered.append(grad_dict)

            # result["newMaliciousClient"] = OrderedDict(grad_list_recovered[f-1])
            # result["isNewMalicoiusClientCalculated"] = True

            values1 = list(grad_list_recovered[len(self.attackerNodes) - 1].values())
            # values2 = list(dict(self.model.state_dict()).values())
            values1 = [v.cpu() for v in values1]

            # values2 = list(dict(self.attackerNodeBeforeAttackWeight[attackerKey]).values())

            attacker_weights = dict(self.model.state_dict())
            values2 = [v for k, v in attacker_weights.items() if k in trainable_parameter]
            difference = [v1 + v2 for v1, v2 in zip(values1, values2)]
            newMaliciousClient = dict(zip(keys, difference))

            # result["newMaliciousClient"] = (newMaliciousClient)
            result["newMaliciousClient"] = OrderedDict(grad_list_recovered[len(self.attackerNodes) - 1])
            result["isNewMalicoiusClientCalculated"] = True

        return result

    def get_all_attacks(self, newMaliciousClient, key, coordinatorNewAttacksLodge, main_short_HoGs, main_long_HoGs, main_normalized_sHoGs, main_full_norm_short_HoGs):
        short_HoGs = deepcopy(main_short_HoGs)
        long_HoGs = deepcopy(main_long_HoGs)
        normalized_sHoGs = deepcopy(main_normalized_sHoGs)
        full_norm_short_HoGs = deepcopy(main_full_norm_short_HoGs)

        avgDect1 = newMaliciousClient
        avgDect2 = self.attackerNodeAvgDelta[key]
        n = self.attackerNodehogAvgLen[key]
        avgValue = OrderedDict({key: (((n* avgDect1[key]) + avgDect2[key]) / (n+1)) for key in avgDect1})
        sHoG = torch.cat([v.flatten() for v in avgValue.values()]).detach().cpu().numpy()
        short_HoGs[key] = sHoG

        sumDect1 = newMaliciousClient
        sumDect2 = self.attackerNodeSumDelta[key]
        sumValue = OrderedDict({key: (sumDect1[key] + sumDect2[key]) for key in sumDect1})
        long_HoGs[key] = torch.cat([v.flatten() for v in sumValue.values()]).detach().cpu().numpy()

        L2_sHoG = np.linalg.norm(sHoG)
        full_norm_short_HoGs.append(sHoG/L2_sHoG)

        if key not in coordinatorNewAttacksLodge["mal_ids"]:
            normalized_sHoGs[key] = sHoG/L2_sHoG

        # STAGE 2 - step 1 : Divating the mud-hug sign flipping detection method
        flip_sign_id = set()
        flip_sign_id = self.get_flipp_sign_attackers(short_HoGs, coordinatorNewAttacksLodge)

        # STAGE 2 - step 2: Divating the mud-hug un targeted detection method
        uAtk_id = set()
        uAtk_id = self.get_additive_noise_attackers(short_HoGs, flip_sign_id, coordinatorNewAttacksLodge)

        # STAGE 2 - step 3: Divating the mud-hug targeted detection method
        offset_tAtk_id1 = set()
        if len(long_HoGs) > 1:
            offset_tAtk_id1 = self.get_targetted_attackers(long_HoGs, flip_sign_id, uAtk_id, coordinatorNewAttacksLodge)

        self.add_mal_id(flip_sign_id, uAtk_id, offset_tAtk_id1, coordinatorNewAttacksLodge)

        unrliable_clients = set()
        unrliable_clients = self.get_unreliable_clients(short_HoGs, flip_sign_id, uAtk_id, offset_tAtk_id1, coordinatorNewAttacksLodge)

        return flip_sign_id, uAtk_id, offset_tAtk_id1, unrliable_clients

    def get_guassian_low_and_high_weight(self, ):
        listOfParamters = {}
        listOfParamtersMean = {}
        listOfParamtersStandardDivation = {}

        numLow = {}
        numHigh = {}

        trainable_parameter = utils.getTrainableParameters(self.model)
        modelData = self.model.state_dict()

        # newMaliciousClient = {}

        for p in self.model.state_dict():
            if p not in trainable_parameter:
                continue
            listOfParamters[p] = []
            listOfParamtersMean[p] = []
            listOfParamtersStandardDivation[p] = []

            for i in self.attackerNodeBeforeAttackWeight:
                malicoiusClient = deepcopy(self.attackerNodeBeforeAttackWeight[i])
                listOfParamters[p].append(malicoiusClient[p])

            arr = np.array(listOfParamters[p])
            listOfParamtersMean[p] = np.mean(arr, axis=0)
            listOfParamtersStandardDivation[p] = np.std(arr, axis=0)

            A = np.array(modelData[p])
            B = np.array(listOfParamtersMean[p])

            if len(list(A.shape))==0:
                A = np.array([modelData[p]])
                B = np.array([listOfParamtersMean[p]])

            cosine = np.sum(A*B)/(norm(A)*norm(B))

            flattened_cosine = cosine.flatten()
            mean_flattened_cosine = flattened_cosine.mean()

            numLow[p] = np.array(listOfParamtersMean[p]) - (4 * np.array(listOfParamtersStandardDivation[p]))
            numHigh[p] = np.array(listOfParamtersMean[p]) - (3 * np.array(listOfParamtersStandardDivation[p]))

            if mean_flattened_cosine < 0:
                numLow[p] = np.array(listOfParamtersMean[p]) + (3 * np.array(listOfParamtersStandardDivation[p]))
                numHigh[p] = np.array(listOfParamtersMean[p]) + (4 * np.array(listOfParamtersStandardDivation[p]))

        return numLow, numHigh

    def get_guassian_low_and_high(self, ):
        listOfParamters = {}
        listOfParamtersMean = {}
        listOfParamtersStandardDivation = {}

        numLow = {}
        numHigh = {}

        trainable_parameter = utils.getTrainableParameters(self.model)
        modelData = self.model.state_dict()

        # newMaliciousClient = {}

        for p in self.model.state_dict():
            if p not in trainable_parameter:
                continue
            listOfParamters[p] = []
            listOfParamtersMean[p] = []
            listOfParamtersStandardDivation[p] = []

            for i in self.attackerNodeBeforeAttackGradient:
                malicoiusClient = deepcopy(self.attackerNodeBeforeAttackGradient[i])
                listOfParamters[p].append(malicoiusClient[p])

            arr = np.array(listOfParamters[p])
            listOfParamtersMean[p] = np.mean(arr, axis=0)
            listOfParamtersStandardDivation[p] = np.std(arr, axis=0)

            A = np.array(modelData[p])
            B = np.array(listOfParamtersMean[p])

            if len(list(A.shape))==0:
                A = np.array([modelData[p]])
                B = np.array([listOfParamtersMean[p]])

            cosine = np.sum(A*B)/(norm(A)*norm(B))

            flattened_cosine = cosine.flatten()
            mean_flattened_cosine = flattened_cosine.mean()

            numLow[p] = np.array(listOfParamtersMean[p]) - (4 * np.array(listOfParamtersStandardDivation[p]))
            numHigh[p] = np.array(listOfParamtersMean[p]) - (3 * np.array(listOfParamtersStandardDivation[p]))

            if mean_flattened_cosine < 0:
                numLow[p] = np.array(listOfParamtersMean[p]) + (3 * np.array(listOfParamtersStandardDivation[p]))
                numHigh[p] = np.array(listOfParamtersMean[p]) + (4 * np.array(listOfParamtersStandardDivation[p]))

        return numLow, numHigh

        #     newSelectedSampledValues = np.full_like(numLow[p], 0)

        #     for index, value in np.ndenumerate(numLow[p]):
        #         rangeStart = value
        #         rangeEnd = numHigh[p][index]

        #         difference = rangeEnd - rangeStart
        #         step = difference / self.samplesThreshold

        #         if step !=0 and (not np.isnan(step)):
        #             tempRange = np.arange(rangeStart, rangeEnd, step)
        #             sampledRangeOfValues = list(tempRange)
        #             flip = np.random.choice([-1, -1])
        #             # random.shuffle(sampledRangeOfValues)
        #             # new_value = sampledRangeOfValues[t] * flip
        #             new_value = sampledRangeOfValues[self.samplesThreshold // 2] * flip
        #             # new_value = value * flip
        #             # new_value = listOfParamters[p][len(self.attackerNodeBeforeAttackGradient)-1][index] * flip
        #             newSelectedSampledValues[index] = new_value
        #         else:
        #             newSelectedSampledValues[index] = value

        #     newMaliciousClient[p] = torch.tensor(newSelectedSampledValues)

        # return newMaliciousClient
    
    def add_mal_id(self, sus_flip_sign, sus_uAtk, sus_tAtk, coordinatorNewAttacksLodge):
        all_suspicious = sus_flip_sign.union(sus_uAtk, sus_tAtk)
        pre_mal_id = coordinatorNewAttacksLodge["pre_mal_id"]
        mal_ids = coordinatorNewAttacksLodge["mal_ids"]
        count_unreliable = coordinatorNewAttacksLodge["count_unreliable"]
        delay_decision = coordinatorNewAttacksLodge["delay_decision"]
        flip_sign_ids = coordinatorNewAttacksLodge["flip_sign_ids"]
        uAtk_ids = coordinatorNewAttacksLodge["uAtk_ids"]
        tAtk_ids = coordinatorNewAttacksLodge["tAtk_ids"]
        unreliable_ids = coordinatorNewAttacksLodge["unreliable_ids"]

        for i in range(self.numberOfClientsTraining):
            if i not in all_suspicious:
                if pre_mal_id[i] == 0:
                    if i in mal_ids:
                        mal_ids.remove(i)
                    if i in flip_sign_ids:
                        flip_sign_ids.remove(i)
                    if i in uAtk_ids:
                        uAtk_ids.remove(i)
                    if i in tAtk_ids:
                        tAtk_ids.remove(i)
                else: #> 0
                    pre_mal_id[i] = 0
                    # Unreliable clients:
                    if i in uAtk_ids:
                        count_unreliable[i] += 1
                        if count_unreliable[i] >= delay_decision:
                            uAtk_ids.remove(i)
                            mal_ids.remove(i)
                            unreliable_ids.add(i)
            else:
                pre_mal_id[i] += 1
                if pre_mal_id[i] >= delay_decision:
                    if i in sus_flip_sign:
                        flip_sign_ids.add(i)
                        mal_ids.add(i)
                    if i in sus_uAtk:
                        uAtk_ids.add(i)
                        mal_ids.add(i)
                if pre_mal_id[i] >= 2*delay_decision and i in sus_tAtk:
                    tAtk_ids.add(i)
                    mal_ids.add(i)
    
    def collect_short_long_hogs(self, ):
        short_HoGs = {}
        long_HoGs = {}
        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        normalized_sHoGs = {}
        full_norm_short_HoGs = [] # for scan flip-sign each round

        for k in range(self.numberOfClientsTraining):
            if k not in self.attackerNodes:
                # shortHoGs
                avgDect1 = self.normalNodeDummyValue[k]
                avgDect2 = self.normalNodeDummyAvgDelta[k]
                n = self.normalNodeDummyhogAvgLen[k]
                avgValue = OrderedDict({key: (((n* avgDect1[key]) + avgDect2[key]) / (n+1)) for key in avgDect1})
                # print(avgDect1)
                # print(avgDect2)
                # print(avgValue)
                sHoG = torch.cat([v.flatten() for v in avgValue.values()]).detach().cpu().numpy()
                # if np.any(np.isnan(sHoG)):
                #     print("Data contains NaN values, consider preprocessing.")
                #     print("short ", k)
                #     print()
                short_HoGs[k] = sHoG
                # longHoGs
                sumDect1 = self.normalNodeDummyValue[k]
                sumDect2 = self.normalNodeDummySumDelta[k]
                sumValue = OrderedDict({key: (sumDect1[key] + sumDect2[key]) for key in sumDect1})
                lHog = torch.cat([v.flatten() for v in sumValue.values()]).detach().cpu().numpy()
                long_HoGs[k] = lHog

                # if np.any(np.isnan(lHog)):
                #     print("Data contains NaN values, consider preprocessing.")
                #     print("long ", k)
                #     print()

                L2_sHoG = np.linalg.norm(sHoG)
                full_norm_short_HoGs.append(sHoG/L2_sHoG)

                # Exclude the firmed malicious clients
                if k not in self.mal_ids:
                    normalized_sHoGs[k] = sHoG/L2_sHoG

        return short_HoGs, long_HoGs, normalized_sHoGs, full_norm_short_HoGs

    def get_flipp_sign_attackers(self, short_HoGs, coordinatorNewAttacksLodge):
        flip_sign_id = set()

        # Pre-convert all short_HoGs to cupy arrays for performance
        short_HoGs_cp = {k: cp.asarray(v) for k, v in short_HoGs.items()}

        # Create a non-malicious copy and compute the median
        non_mal_sHoGs_cp = {k: v for k, v in short_HoGs_cp.items() if k not in coordinatorNewAttacksLodge["mal_ids"]}

        # Stack into a 2D array and compute median
        median_sHoG = cp.median(cp.stack(list(non_mal_sHoGs_cp.values())), axis=0)

        # Cosine similarity and sign flipping check
        for i, v in short_HoGs_cp.items():
            dot = cp.dot(median_sHoG, v)
            norm_product = cp.linalg.norm(median_sHoG) * cp.linalg.norm(v)
            
            if norm_product != 0:
                d_cos = dot / norm_product
                if d_cos < 0:
                    flip_sign_id.add(i)
        # # print(short_HoGs.keys())
        # # print(coordinatorNewAttacksLodge["mal_ids"])
        # flip_sign_id = set()
        # non_mal_sHoGs = dict(short_HoGs) # deep copy dict
        # for i in coordinatorNewAttacksLodge["mal_ids"]:
        #     if i in non_mal_sHoGs:
        #        non_mal_sHoGs.pop(i)
        # # print(non_mal_sHoGs.keys())
        # # print(len(non_mal_sHoGs))
        # # print(len(non_mal_sHoGs.values()))
        # # print(non_mal_sHoGs.keys())
        # # print(type(non_mal_sHoGs))

        # # print(len(short_HoGs))
        # # print(len(short_HoGs.values()))
        # # print(short_HoGs.keys())
        # # print(type(short_HoGs))

        # # for key, value in non_mal_sHoGs.items():
        # #     print(f"Key: {key}, Size: {len(value)}")

        # # for key, value in short_HoGs.items():
        # #     print(f"Key short: {key}, Size: {len(value)}")

        # median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
        # # print("attack calcalated median for sign flipping", median_sHoG)
        # for i, v in short_HoGs.items():
        #     v = np.array(list(v))
        #     if (np.linalg.norm(median_sHoG)*np.linalg.norm(v)) != 0:
        #         d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
        #         if d_cos < 0: # angle > 90
        #             flip_sign_id.add(i)

        return flip_sign_id

    def get_additive_noise_attackers(self, short_HoGs, flip_sign_id, coordinatorNewAttacksLodge):
        # print(4)
        for i in range(self.numberOfClientsTraining):
            if i in flip_sign_id or i in coordinatorNewAttacksLodge["flip_sign_ids"]:
                if i in short_HoGs:
                    short_HoGs.pop(i)
        id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))

        # print(44)

        value_sHoGs = np.clip(value_sHoGs, -1e10, 1e10)
        if np.any(np.isnan(value_sHoGs)):
            # print("imputer")
            imputer = SimpleImputer(strategy='mean')  # Other options: 'median', 'most_frequent', 'constant'
            value_sHoGs = imputer.fit_transform(value_sHoGs)

        # print(4444)

        sus_tAtk_uRel_id0, uAtk_id = set(), set()

        # print(value_sHoGs.shape)
        # print(len(value_sHoGs[0]))


        if value_sHoGs.shape[1] !=0:

            # svd = TruncatedSVD(n_components=500, random_state=42)
            # reduced_features = svd.fit_transform(value_sHoGs)

            # cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
            #     min_samples=self.dbscan_min_samples).fit(reduced_features)
            
            cluster_sh = cuDBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            cluster_sh.fit(value_sHoGs)
            
            # print(444444)
            
            offset_normal_ids = self.find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)

            # print(4444444444)

            # print("attack calcalated median for untargetted", normal_cent)

            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]

            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids
            d_normal_sus = {} # distance from centroid of normal to suspicious clients.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)


            # print(444444444444444)

            d_separate = self.find_separate_point(list(d_normal_sus.values()))
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(int(k))
                else:
                    sus_tAtk_uRel_id0.add(int(k))

            # print(444444444444444444444444444)

        return uAtk_id
    
    def get_targetted_attackers(self, long_HoGs, flip_sign_id, uAtk_ids, coordinatorNewAttacksLodge):
        for i in range(self.numberOfClientsTraining):
            if i in coordinatorNewAttacksLodge["flip_sign_ids"] or i in flip_sign_id:
                if i in long_HoGs:
                    long_HoGs.pop(i)
            if i in uAtk_ids or i in coordinatorNewAttacksLodge["uAtk_ids"]:
                if i in long_HoGs:
                    long_HoGs.pop(i)
        id_lHoGs = np.array(list(long_HoGs.keys()))
        value_lHoGs = np.array(list(long_HoGs.values()))

        value_lHoGs = np.clip(value_lHoGs, -1e10, 1e10)

        if np.any(np.isnan(value_lHoGs)):
            imputer = SimpleImputer(strategy='mean')  # Other options: 'median', 'most_frequent', 'constant'
            value_lHoGs = imputer.fit_transform(value_lHoGs)

        offset_tAtk_id1 = {}

        if value_lHoGs.shape[1]!=0:
            cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
            offset_tAtk_id1 = self.find_minority_id(cluster_lh1)
            # sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
            offset_tAtk_id1 = {int(x) for x in offset_tAtk_id1}

        return offset_tAtk_id1

    def get_unreliable_clients(self, short_HoGs, flip_sign_id, uAtk_ids, offset_tAtk_id1, coordinatorNewAttacksLodge):
        for i in range(self.numberOfClientsTraining):
            if i in flip_sign_id or i in uAtk_ids or i in offset_tAtk_id1:
                if i in short_HoGs:
                    short_HoGs.pop(i)

        angle_sHoGs = {}
        # update this value again after excluding malicious clients
        median_sHoG = np.median(np.array(list(short_HoGs.values())), axis=0)
        # print("attack calcalated median for unreliable", median_sHoG)
        for i, v in short_HoGs.items():
            angle_sHoGs[i] = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))

        angle_sep_sH = self.find_separate_point(list(angle_sHoGs.values()))
        normal_id, uRel_id = set(), set()
        for k, v in angle_sHoGs.items():
            if v < angle_sep_sH: # larger angle, smaller cosine similarity
                uRel_id.add(k)
            else:
                normal_id.add(k)

        count_unreliable = coordinatorNewAttacksLodge["count_unreliable"]
        delay_decision = coordinatorNewAttacksLodge["delay_decision"]
        uAtk_ids = coordinatorNewAttacksLodge["uAtk_ids"]
        unreliable_ids = coordinatorNewAttacksLodge["unreliable_ids"]

        for k in range(self.numberOfClientsTraining):
                if k in uRel_id:
                    count_unreliable[k] += 1
                    if count_unreliable[k] > delay_decision:
                        unreliable_ids.add(k)
                # do this before decreasing count
                if count_unreliable[k] == 0 and k in unreliable_ids:
                    unreliable_ids.remove(k)
                if k not in uRel_id and count_unreliable[k] > 0:
                    count_unreliable[k] -= 1

        return uRel_id

    def calculate_attacks_score(self, key, coordinatorNewAttacksLodge):
        tAtk_ids = coordinatorNewAttacksLodge["tAtk_ids"]
        uAtk_ids = coordinatorNewAttacksLodge["uAtk_ids"]
        flip_sign_ids = coordinatorNewAttacksLodge["flip_sign_ids"]
        unreliable_ids = coordinatorNewAttacksLodge["unreliable_ids"]

        intersection = tAtk_ids & uAtk_ids & flip_sign_ids & unreliable_ids
        union = tAtk_ids | uAtk_ids | flip_sign_ids | unreliable_ids

        inter1 = tAtk_ids & uAtk_ids
        inter2 = tAtk_ids & flip_sign_ids
        inter3 = tAtk_ids & unreliable_ids
        inter4 = uAtk_ids & flip_sign_ids
        inter5 = uAtk_ids & unreliable_ids
        inter6 = flip_sign_ids & unreliable_ids

        inter7 = tAtk_ids & uAtk_ids & flip_sign_ids
        inter8 = tAtk_ids & uAtk_ids & unreliable_ids
        inter9 = tAtk_ids & flip_sign_ids & unreliable_ids
        inter10 = uAtk_ids & flip_sign_ids & unreliable_ids

        unique_set1 = tAtk_ids - (uAtk_ids | flip_sign_ids | unreliable_ids)
        unique_set2 = uAtk_ids - (tAtk_ids | flip_sign_ids | unreliable_ids)
        unique_set3 = flip_sign_ids - (tAtk_ids | uAtk_ids | unreliable_ids)
        unique_set4 = unreliable_ids - (flip_sign_ids | tAtk_ids | uAtk_ids)

        keysToDiscard = set(self.attackerNodeOfLabelFlipping) | set(self.attackerNodeOfBackdoor) | set(self.attackerNodeOfMultiLabelFlipping) | {key}

        union.difference_update(keysToDiscard)
        
        score = len(union)

        intersection.difference_update(keysToDiscard)
        inter1.difference_update(keysToDiscard)
        inter2.difference_update(keysToDiscard)
        inter3.difference_update(keysToDiscard)
        inter4.difference_update(keysToDiscard)
        inter5.difference_update(keysToDiscard)
        inter6.difference_update(keysToDiscard)
        inter7.difference_update(keysToDiscard)
        inter8.difference_update(keysToDiscard)
        inter9.difference_update(keysToDiscard)
        inter10.difference_update(keysToDiscard)

        twoIntersectionScoreMultiplier = 2
        threeIntersectionScoreMultiplier = 3
        fourIntersectionScoreMultiplier = 4

        score+=len(inter1) * twoIntersectionScoreMultiplier
        score+=len(inter2) * twoIntersectionScoreMultiplier
        score+=len(inter3) * twoIntersectionScoreMultiplier
        score+=len(inter4) * twoIntersectionScoreMultiplier
        score+=len(inter5) * twoIntersectionScoreMultiplier
        score+=len(inter6) * twoIntersectionScoreMultiplier
        score+=len(inter7) * threeIntersectionScoreMultiplier
        score+=len(inter8) * threeIntersectionScoreMultiplier
        score+=len(inter9) * threeIntersectionScoreMultiplier
        score+=len(inter10) * threeIntersectionScoreMultiplier
        score+=len(intersection) * fourIntersectionScoreMultiplier

        keysFoundScoreMultiplier = 4

        score -= len(keysToDiscard & flip_sign_ids) * keysFoundScoreMultiplier
        score -= len(keysToDiscard & uAtk_ids) * keysFoundScoreMultiplier
        score -= len(keysToDiscard & tAtk_ids) * keysFoundScoreMultiplier
        score -= len(keysToDiscard & unreliable_ids) * keysFoundScoreMultiplier

        # if key in coordinatorNewAttacksLodge["flip_sign_ids"]:
        #     score-=4

        # if key in coordinatorNewAttacksLodge["uAtk_ids"]:
        #     score-=4
        
        # if key in coordinatorNewAttacksLodge["tAtk_ids"]:
        #     score-=4

        # if key in coordinatorNewAttacksLodge["unreliable_ids"]:
        #     score-=4

        # print("multi label flipping", self.attackerNodeOfMultiLabelFlipping)
        # print("label flipping", self.attackerNodeOfLabelFlipping)
        # print("back door", self.attackerNodeOfBackdoor)
        # print("pre_mal_id", coordinatorNewAttacksLodge["pre_mal_id"])
        # print("global flip_sign_ids", coordinatorNewAttacksLodge["flip_sign_ids"])
        # print("global uAtk_ids", coordinatorNewAttacksLodge["uAtk_ids"])
        # print("global tAtk_ids", coordinatorNewAttacksLodge["tAtk_ids"])
        # print("global unreliable_ids", coordinatorNewAttacksLodge["unreliable_ids"])
        # print("keysToDiscard", keysToDiscard)
        # print("score", score)
        # print("tAtk_ids", tAtk_ids)
        # print("uAtk_id", uAtk_ids)
        # print("flip_sign_id", flip_sign_ids)
        # print("unreliable_clients", unreliable_ids)
        # print(len(tAtk_ids))
        # print(len(uAtk_ids))
        # print(len(flip_sign_ids))
        # print(len(unreliable_ids))
        # print(f"Intersection: {intersection}")
        # print(f"Union: {union}")
        # print(len(intersection))
        # print(len(union))
        # print("key", key)
        # print(f"intersection1: {inter1}")
        # print(f"intersection2: {inter2}")
        # print(f"intersection3: {inter3}")
        # print(f"intersection4: {inter4}")
        # print(f"intersection5: {inter5}")
        # print(f"intersection6: {inter6}")
        # print(f"intersection7: {inter7}")
        # print(f"intersection8: {inter8}")
        # print(f"intersection9: {inter9}")
        # print(f"intersection10: {inter10}")
        # print(f"Unique in set1: {unique_set1}")
        # print(f"Unique in set2: {unique_set2}")
        # print(f"Unique in set3: {unique_set3}")
        # print(f"Unique in set4: {unique_set4}")
        # print()

        return score

    def find_minority_id(self, clf):
        count_1 = sum(clf.labels_ == 1)
        count_0 = sum(clf.labels_ == 0)
        mal_label = 0 if count_1 > count_0 else 1
        atk_id = np.where(clf.labels_ == mal_label)[0]
        atk_id = set(atk_id.reshape((-1)))
        return atk_id
    
    def find_majority_id(self, clf):
        labels = cp.asnumpy(clf.labels_)
        # labels = clf.labels_
        counts = Counter(labels)
        major_label = max(counts, key=counts.get)
        major_id = np.where(clf.labels_ == major_label)[0]
        #major_id = set(major_id.reshape(-1))
        return major_id

    def find_separate_point(self, d):
        # d should be flatten and np or list
        d = sorted(d)
        sep_point = 0
        max_gap = 0
        for i in range(len(d)-1):
            if d[i+1] - d[i] > max_gap:
                max_gap = d[i+1] - d[i]
                sep_point = d[i] + max_gap/2
        return sep_point
