from __future__ import print_function

import random
import statistics
from numpy.linalg import norm
from collections import OrderedDict,Counter,defaultdict
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


class SignFlibbingMaliciousCordinator():
    signFlippingAttackerBeforeAttackGradient = {}
    signFlippingAttackerAvgDelta = {}
    signFlippingAttackerSumDelta = {}
    signFlippingAttackerhogAvgLen = {}
    normalClientDummyValue = {}
    normalClientDummyAvgDelta = {}
    normalClientDummySumDelta = {}
    normalClientDummyhogAvgLen = {}
    signFlippingAttackers = []
    numberOfClientsTraining = 0
    numberOfSignFlippingAttackers = 0
    model = None
    minimumNumberOfMaliciousClientsThreshold = 0
    minimumIterationToStartTheAttack = 0

    samplesThreshold = 5

    dbscan_eps = 0.5
    dbscan_min_samples = 5

    signFlippingChoosenGradientInThisIteration = {}
    signFlippingChoosenAvgDeltaInThisIteration = {}
    signFlippingChoosenSumDeltaInThisIteration = {}
    signFlippingChoosenhogAvgLenInThisIteration = {}
    currentIteration = 0


    def addMaliciousClientUpdate(client):
        SignFlibbingMaliciousCordinator.signFlippingAttackers.append(client.cid)
        SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers += 1

        trainable_parameter = utils.getTrainableParameters(SignFlibbingMaliciousCordinator.model)

        clientState = client.model.state_dict()
        clientAvgDelta = client.avg_delta
        clientSumDelta = client.sum_hog
        clientStateChange = client.model.state_dict()

        for p in client.originalState:
            if p not in trainable_parameter:
                continue
            clientStateChange[p] = clientState[p] - client.originalState[p]

        SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient[client.cid] = deepcopy(clientStateChange)
        SignFlibbingMaliciousCordinator.signFlippingAttackerAvgDelta[client.cid] = deepcopy(clientAvgDelta)
        SignFlibbingMaliciousCordinator.signFlippingAttackerSumDelta[client.cid] = deepcopy(clientSumDelta)
        SignFlibbingMaliciousCordinator.signFlippingAttackerhogAvgLen[client.cid] = len(client.hog_avg)

        result = {
            "newMaliciousClient": None,
            "isNewMalicoiusClientCalculated": False,
        }

        modelData = SignFlibbingMaliciousCordinator.model.state_dict()

        if len(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient) == SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers and SignFlibbingMaliciousCordinator.currentIteration>=SignFlibbingMaliciousCordinator.minimumIterationToStartTheAttack and SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers>=SignFlibbingMaliciousCordinator.minimumNumberOfMaliciousClientsThreshold:
            dict1 = dict(SignFlibbingMaliciousCordinator.signFlippingChoosenGradientInThisIteration)
            dict2 = SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(SignFlibbingMaliciousCordinator.signFlippingChoosenAvgDeltaInThisIteration)
            dict2 = SignFlibbingMaliciousCordinator.signFlippingAttackerAvgDelta
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(SignFlibbingMaliciousCordinator.signFlippingChoosenSumDeltaInThisIteration)
            dict2 = SignFlibbingMaliciousCordinator.signFlippingAttackerSumDelta
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            dict1 = dict(SignFlibbingMaliciousCordinator.signFlippingChoosenhogAvgLenInThisIteration)
            dict2 = SignFlibbingMaliciousCordinator.signFlippingAttackerhogAvgLen
            dict2.update({k: v for k, v in dict1.items() if k in dict2})

            SignFlibbingMaliciousCordinator.normalClientDummyValue = {}
            SignFlibbingMaliciousCordinator.normalClientDummyAvgDelta = {}
            SignFlibbingMaliciousCordinator.normalClientDummySumDelta = {}
            SignFlibbingMaliciousCordinator.normalClientDummyhogAvgLen = {}
            
            listOfMaliciousClientsIdsToChooseFrom = list(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient.keys())

            for i in range(SignFlibbingMaliciousCordinator.numberOfClientsTraining):
                if i not in SignFlibbingMaliciousCordinator.signFlippingAttackers:
                    selectedRandomMaliciousKey = random.sample(listOfMaliciousClientsIdsToChooseFrom, k=1)[0]
                    SignFlibbingMaliciousCordinator.normalClientDummyValue[i] = (SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient[selectedRandomMaliciousKey])
                    SignFlibbingMaliciousCordinator.normalClientDummyAvgDelta[i] = (SignFlibbingMaliciousCordinator.signFlippingAttackerAvgDelta[selectedRandomMaliciousKey])
                    SignFlibbingMaliciousCordinator.normalClientDummySumDelta[i] = (SignFlibbingMaliciousCordinator.signFlippingAttackerSumDelta[selectedRandomMaliciousKey])
                    SignFlibbingMaliciousCordinator.normalClientDummyhogAvgLen[i] = (SignFlibbingMaliciousCordinator.signFlippingAttackerhogAvgLen[selectedRandomMaliciousKey])

            listOfParamters = {}
            listOfParamtersMean = {}
            listOfParamtersStandardDivation = {}
            newMaliciousClient = {}

            # TODO stable point 1

            listOfNewMaliciousClient = {}
            listOfNumberOfClientMarkedAsMalicious = {}
            listOfClientsMarkedAsMalicious = {}

            for t in range(SignFlibbingMaliciousCordinator.samplesThreshold):
                for p in SignFlibbingMaliciousCordinator.model.state_dict():
                    if p not in trainable_parameter:
                        continue
                    listOfParamters[p] = []
                    listOfParamtersMean[p] = []
                    listOfParamtersStandardDivation[p] = []

                    for i in SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient:
                        malicoiusClient = (SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient[i])
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

                    numLow = np.array(listOfParamtersMean[p]) - (4 * np.array(listOfParamtersStandardDivation[p]))
                    numHigh = np.array(listOfParamtersMean[p]) - (3 * np.array(listOfParamtersStandardDivation[p]))

                    if mean_flattened_cosine < 0:
                        numLow = np.array(listOfParamtersMean[p]) + (3 * np.array(listOfParamtersStandardDivation[p]))
                        numHigh = np.array(listOfParamtersMean[p]) + (4 * np.array(listOfParamtersStandardDivation[p]))

                    newSelectedSampledValues = np.full_like(numLow, 0)

                    for index, value in np.ndenumerate(numLow):
                        rangeStart = value
                        rangeEnd = numHigh[index]

                        difference = rangeEnd - rangeStart
                        step = difference / SignFlibbingMaliciousCordinator.samplesThreshold

                        if step !=0 and (not np.isnan(step)):
                            tempRange = np.arange(rangeStart, rangeEnd, step)
                            sampledRangeOfValues = list(tempRange)
                            flip = np.random.choice([-1, -1])
                            # random.shuffle(sampledRangeOfValues)
                            new_value = sampledRangeOfValues[t] * flip
                            # new_value = value * flip
                            # new_value = listOfParamters[p][len(SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient)-1][index] * flip
                            newSelectedSampledValues[index] = new_value
                        else:
                            newSelectedSampledValues[index] = value

                    newMaliciousClient[p] = torch.tensor(newSelectedSampledValues)

                # STAGE 1: Collect long and short HoGs.
                short_HoGs = {}
                long_HoGs = {}
                key = None

                short_HoGs, long_HoGs, key = SignFlibbingMaliciousCordinator.collect_short_long_hogs(newMaliciousClient)

                # STAGE 2 : Divating the mud-hug sign flipping detection method
                flip_sign_id = set()
                flip_sign_id = SignFlibbingMaliciousCordinator.get_flipp_sign_attackers(short_HoGs)

                # STAGE 3: Divating the mud-hug un targeted detection method
                uAtk_id = set()
                uAtk_id = SignFlibbingMaliciousCordinator.get_additive_noise_attackers(short_HoGs)

                # STAGE 4: Divating the mud-hug targeted detection method
                offset_tAtk_id1 = set()
                if len(long_HoGs) > 1:
                    offset_tAtk_id1 = SignFlibbingMaliciousCordinator.get_targetted_attackers(long_HoGs)

                # STAGE 5: Calcualte best score
                listOfNumberOfClientMarkedAsMalicious[t] = SignFlibbingMaliciousCordinator.calculate_attacks_score(offset_tAtk_id1, uAtk_id, flip_sign_id, key)
                listOfNewMaliciousClient[t] = OrderedDict(newMaliciousClient)

                # print(listOfNumberOfClientMarkedAsMalicious[t])
                # print(t)
                # print()

            
            items = list(listOfNumberOfClientMarkedAsMalicious.items())
            larget_value_index = 0
            for i in range(len(items)):
                larget_value_index = i
                for j in range(i + 1, len(items)):
                    if items[j][1] >= items[larget_value_index][1]:  # Compare values
                        larget_value_index = j

            # for i in listOfNumberOfClientMarkedAsMalicious:
            #     print("threshold", i)
            #     print("total detected", listOfNumberOfClientMarkedAsMalicious[i])
            #     print()

            result["newMaliciousClient"] = listOfNewMaliciousClient[larget_value_index]
            result["isNewMalicoiusClientCalculated"] = True
            # TODO stable point 2
        return result
    
    def collect_short_long_hogs(newMaliciousClient):
        short_HoGs = {}
        long_HoGs = {}
        for k in range(SignFlibbingMaliciousCordinator.numberOfClientsTraining):
            if k not in SignFlibbingMaliciousCordinator.signFlippingAttackers:
                # shortHoGs
                avgDect1 = SignFlibbingMaliciousCordinator.normalClientDummyValue[k]
                avgDect2 = SignFlibbingMaliciousCordinator.normalClientDummyAvgDelta[k]
                n = SignFlibbingMaliciousCordinator.normalClientDummyhogAvgLen[k]
                avgValue = OrderedDict({key: (((n* avgDect1[key]) + avgDect2[key]) / (n+1)) for key in avgDect1})
                short_HoGs[k] = torch.cat([v.flatten() for v in avgValue.values()]).detach().cpu().numpy()
                # longHoGs
                sumDect1 = SignFlibbingMaliciousCordinator.normalClientDummyValue[k]
                sumDect2 = SignFlibbingMaliciousCordinator.normalClientDummySumDelta[k]
                sumValue = OrderedDict({key: (sumDect1[key] + sumDect2[key]) for key in sumDect1})
                long_HoGs[k] = torch.cat([v.flatten() for v in sumValue.values()]).detach().cpu().numpy()

        keys = SignFlibbingMaliciousCordinator.signFlippingAttackerBeforeAttackGradient.keys()
        key = list(keys)[SignFlibbingMaliciousCordinator.numberOfSignFlippingAttackers-1]
        avgDect1 = newMaliciousClient
        avgDect2 = SignFlibbingMaliciousCordinator.signFlippingAttackerAvgDelta[key]
        n = SignFlibbingMaliciousCordinator.signFlippingAttackerhogAvgLen[key]
        avgValue = OrderedDict({key: (((n* avgDect1[key]) + avgDect2[key]) / (n+1)) for key in avgDect1})
        short_HoGs[key] = torch.cat([v.flatten() for v in avgValue.values()]).detach().cpu().numpy()

        sumDect1 = newMaliciousClient
        sumDect2 = SignFlibbingMaliciousCordinator.signFlippingAttackerSumDelta[key]
        sumValue = OrderedDict({key: (sumDect1[key] + sumDect2[key]) for key in sumDect1})
        long_HoGs[key] = torch.cat([v.flatten() for v in sumValue.values()]).detach().cpu().numpy()

        return short_HoGs, long_HoGs, key

    def get_flipp_sign_attackers(short_HoGs):
        flip_sign_id = set()
        non_mal_sHoGs = dict(short_HoGs)
        median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
        for i, v in short_HoGs.items():
            v = np.array(list(v))
            if (np.linalg.norm(median_sHoG)*np.linalg.norm(v)) != 0:
                d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                if d_cos < 0: # angle > 90
                    flip_sign_id.add(i)
        
        return flip_sign_id

    def get_additive_noise_attackers(short_HoGs):
        id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))

        start_t = time.time()

        cluster_sh = DBSCAN(eps=SignFlibbingMaliciousCordinator.dbscan_eps, n_jobs=-1,
            min_samples=SignFlibbingMaliciousCordinator.dbscan_min_samples).fit(value_sHoGs)
        
        offset_normal_ids = SignFlibbingMaliciousCordinator.find_majority_id(cluster_sh)
        normal_ids = id_sHoGs[list(offset_normal_ids)]
        normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
        normal_cent = np.median(normal_sHoGs, axis=0)
        offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
        sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]

        suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids
        d_normal_sus = {} # distance from centroid of normal to suspicious clients.
        for sid in suspicious_ids:
            d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)

        d_separate = SignFlibbingMaliciousCordinator.find_separate_point(list(d_normal_sus.values()))
        sus_tAtk_uRel_id0, uAtk_id = set(), set()
        for k, v in d_normal_sus.items():
            if v > d_separate and k in sus_uAtk_ids:
                uAtk_id.add(int(k))
            else:
                sus_tAtk_uRel_id0.add(int(k))

        return uAtk_id
    
    def get_targetted_attackers(long_HoGs):
        id_lHoGs = np.array(list(long_HoGs.keys()))
        value_lHoGs = np.array(list(long_HoGs.values()))

        cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
        offset_tAtk_id1 = SignFlibbingMaliciousCordinator.find_minority_id(cluster_lh1)
        # sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
        offset_tAtk_id1 = {int(x) for x in offset_tAtk_id1}
        return offset_tAtk_id1

    
    def calculate_attacks_score(offset_tAtk_id1, uAtk_id, flip_sign_id, key):
        intersection = offset_tAtk_id1 & uAtk_id & flip_sign_id
        union = offset_tAtk_id1 | uAtk_id | flip_sign_id
        intersection1 = offset_tAtk_id1 & uAtk_id
        intersection2 = uAtk_id & flip_sign_id
        intersection3 = offset_tAtk_id1 & flip_sign_id
        unique_set1 = offset_tAtk_id1 - (uAtk_id | flip_sign_id)
        unique_set2 = uAtk_id - (offset_tAtk_id1 | flip_sign_id)
        unique_set3 = flip_sign_id - (offset_tAtk_id1 | uAtk_id)

        # print(offset_tAtk_id1)
        # print(uAtk_id)
        # print(flip_sign_id)
        # print(len(offset_tAtk_id1))
        # print(len(uAtk_id))
        # print(len(flip_sign_id))
        # print(f"Intersection: {intersection}")
        # print(f"Union: {union}")
        # print(len(intersection))
        # print(len(union))
        # print("key", key)
        # print(f"intersection1: {intersection1}")
        # print(f"intersection2: {intersection2}")
        # print(f"intersection3: {intersection3}")
        # print(f"Unique in set1: {unique_set1}")
        # print(f"Unique in set2: {unique_set2}")
        # print(f"Unique in set3: {unique_set3}")
        # print()

        listOfNumberOfClientMarkedAsMalicious = len(union)

        if key in union:
            listOfNumberOfClientMarkedAsMalicious-=1

        intersection.discard(key)
        intersection1.discard(key)
        intersection2.discard(key)
        intersection3.discard(key)

        listOfNumberOfClientMarkedAsMalicious+=len(intersection) * 3
        listOfNumberOfClientMarkedAsMalicious+=len(intersection1) * 2
        listOfNumberOfClientMarkedAsMalicious+=len(intersection2) * 2
        listOfNumberOfClientMarkedAsMalicious+=len(intersection3) * 2

        return listOfNumberOfClientMarkedAsMalicious

    def find_minority_id(clf):
        count_1 = sum(clf.labels_ == 1)
        count_0 = sum(clf.labels_ == 0)
        mal_label = 0 if count_1 > count_0 else 1
        atk_id = np.where(clf.labels_ == mal_label)[0]
        atk_id = set(atk_id.reshape((-1)))
        return atk_id
    
    
    def find_majority_id(clf):
        counts = Counter(clf.labels_)
        major_label = max(counts, key=counts.get)
        major_id = np.where(clf.labels_ == major_label)[0]
        #major_id = set(major_id.reshape(-1))
        return major_id

    def find_separate_point(d):
        # d should be flatten and np or list
        d = sorted(d)
        sep_point = 0
        max_gap = 0
        for i in range(len(d)-1):
            if d[i+1] - d[i] > max_gap:
                max_gap = d[i+1] - d[i]
                sep_point = d[i] + max_gap/2
        return sep_point
