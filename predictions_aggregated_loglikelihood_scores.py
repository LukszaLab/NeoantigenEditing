#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script for comparing fitness models and computing log-likelihood scores
    Copyright (C) 2022 Marta Luksza
"""

import glob
import json
import os
import numpy as np
import pandas as pd


def set_fitness_on_a_tree(tree, weights):
    '''
    Computes fitness of all clones in a tree as a weighted sum of fitness components,
    F_I and F_P

    :param tree: dict
        json representation of a tree

    :param weights: dict
        {"Sigma_I" -> float, "Sigma_P"->float}
    '''
    nodes = [tree["topology"]]
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "predicted_x" in node:
            del node["predicted_x"]
        if "predicted_X" in node:
            del node["predicted_X"]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        node["fitness"] = node["F_I"] * weights["Sigma_I"] + node["F_P"] * weights["Sigma_P"]


def path2root(nid, parent):
    if nid == 0:
        return [nid]
    else:
        rpath = [nid] + path2root(parent[nid], parent)
        return rpath

def get_property(tree, property):
    '''
    Auxiliary function to extract a clone attribute values into a list
    :param tree: dict
        json representation of a tree

    :param property: str
        name of the attribute

    :return list
    '''
    nodes = [tree["topology"]]
    vals = []
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        vals.append((node["clone_id"], node[property]))
    vals.sort(key=lambda x: x[0])
    vals = [x for (_, x) in vals]
    return vals


def KL_null_tree(prim_tree, rec_tree):
    '''
    Computes Kullback-Leibler divergence between the recurrent and the primary tumor tree.
    The topology of the clonal structure tree is common between the two time points,
    the probability distributions are defined by clone frequencies that differ between
    the two time points.

    :param prim_tree: dict
        json representation of a tree (primary tumor)

    :param rec_tree: dict
        json representation of a tree (recurrent tumor)

    :return float:
        Kullback-Leibler divergence value
    '''
    x0 = get_property(prim_tree, "tilde_x")
    observed_x = get_property(rec_tree, "tilde_x")
    pairs = list(zip(x0, observed_x))
    pairs = [x for x in pairs if x[1] > 0 and x[0] > 0]

    kl = sum(obs_x * np.log(obs_x / null_x) for (null_x, obs_x) in pairs)
    return kl


def KL_prediction_tree(prim_tree, rec_tree):
    '''
    Computes Kullback-Leibler divergence between the recurrent tumor tree and
    the tree predicted with the fitness model applied on the primary tree.
    The topology of the clonal structure tree is common between the two time points,
    the probability distributions are defined by the observed and predicted clone frequencies.

    :param prim_tree: dict
        json representation of a tree (primary tumor)

    :param rec_tree: dict
        json representation of a tree (recurrent tumor)

    :return float:
        Kullback-Leibler divergence value
    '''
    x0 = get_property(prim_tree, "tilde_x")
    observed_x = get_property(rec_tree, "tilde_x")
    predicted_x = get_property(prim_tree, "predicted_x")
    pairs = list(zip(predicted_x, observed_x, x0))
    pairs = [x for x in pairs if x[2] > 0 and x[1] > 0]

    kl = sum(obs_x * np.log(obs_x / pred_x) for (pred_x, obs_x, _) in pairs)
    return kl


def KL_prediction(prim_json, rec_json):
    '''
    Kullback-Leibler divergence between the observed and predicted recurrent tumor sample.
    The divergence is computed as a weighted sum over divergences between 5 top scoring trees.
    Weights are proportional to the likelihood of the trees.

    :param prim_json: dict
        json representation of the primary tumor sample

    :param rec_json: dict
        json representation of the recurrent tumor sample

    :return float
        Kullback-Leibler divergence between the observed and predicted clone compositions
        of the recurrent tumor.
    '''
    prim_trees = prim_json["sample_trees"]
    rec_trees = rec_json["sample_trees"]
    weights = np.array([tree["score"] for tree in prim_json["sample_trees"]])
    weights = np.exp(weights - max(weights))
    weights = weights / sum(weights)
    kl = sum([w * KL_prediction_tree(prim_tree, rec_tree) for (prim_tree, rec_tree, w) in
              zip(prim_trees, rec_trees, weights)])
    return kl


def KL_null(prim_json, rec_json):
    '''
    Kullback-Leibler divergence between the primary and recurrent tumor samples. Computed
    as the weighted average over the top 5 highest scoring clonal tree structures.

    :param prim_json: dict
        json representation of the primary tumor sample

    :param rec_json: dict
        json representation of the recurrent tumor sample

    :return float
        Kullback-Leibler divergence between the clone compositions
        of the primary and the recurrent tumor.

    '''
    prim_trees = prim_json["sample_trees"]
    rec_trees = rec_json["sample_trees"]
    weights = np.array([tree["score"] for tree in prim_json["sample_trees"]])
    weights = np.exp(weights - max(weights))
    weights = weights / sum(weights)
    kl = sum(
        [w * KL_null_tree(prim_tree, rec_tree) for (prim_tree, rec_tree, w) in zip(prim_trees, rec_trees, weights)])
    return kl

def predict_on_a_tree(tree):
    '''
    Applies fitness model projection on the primary tumor tree. Computes "predicted_x"
    attribute for each clone in the tree.
    '''
    nodes = [tree["topology"]]
    Z = 0.0
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        node["predicted_x"] = node["tilde_x"] * np.exp(node["fitness"])
        Z += node["predicted_x"]

    # normalize
    nodes = [tree["topology"]]
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        node["predicted_x"] /= Z


def predict(tumor_json, weights):
    '''
    Computes fitness of all clones and performs clone composition predictions for
    all trees.

    :param tumor_json: dict
        json representation of a sample

    :param weights: dict
        fitness component weights, Sigma_I and Sigma_P

    '''
    for tree in tumor_json["sample_trees"]:
        set_fitness_on_a_tree(tree, weights)
        predict_on_a_tree(tree)


def optimize_weights(prim_json, rec_json):
    '''
    Optimize fitness model component weights, Sigma_I and Sigma_P
    to minimize the Kullback-Leibler divergence between the predicted and observed recurrent
    tumor frequencies.

    :param prim_json: dict
        primary tumor json

    :param rec_json: dict
        recurrent tumor json

    :return dict:
        "Sigma_I" -> float
        "Sigma_P" -> float
    '''
    from scipy.optimize import basinhopping
    from scipy.optimize import LinearConstraint

    def distance_fun(x, *args):
        '''

        :param x: tuple of floats
            (sigma_I, sigma_P)
            parameters to be optimized

        :param args: list
            (prim_json, tumor_json)

        :return: float
        '''

        prim_json = args[0]
        rec_json = args[1]
        weights = {"Sigma_I": x[0], "Sigma_P": x[1]}
        predict(prim_json, weights)
        prediction_KL = KL_prediction(prim_json, rec_json)
        return prediction_KL

    x0 = (0,0)
    diag = np.diag([1,1])
    linear_constraint = LinearConstraint(diag, 0, 10) #weights have to be >= 0

    minkwargs = {"args": (prim_json, rec_json),
                 "constraints": linear_constraint}
    val = basinhopping(distance_fun, x0, niter=500,
                       minimizer_kwargs=minkwargs)
    opt_weights = {"Sigma_I": val["x"][0], "Sigma_P": val["x"][1]}
    return opt_weights

if __name__ == "__main__":

    '''
    Performs clone frequency predictions
     
    '''

    dir = os.path.join("data")
    patient_dir = os.path.join("data", "Patient_data")

    patdirs = [x for x in glob.glob(os.path.join("data", "Patient_data", "*")) if os.path.isdir(x)]

    aggregated_LL = {"LTS": 0, "STS": 0}
    lls = {"LTS": [], "STS": []}

    for pat in patdirs:
        primdir = os.path.join(pat, "Primary")
        recdir = os.path.join(pat, "Recurrent")
        rec_tumors = [os.path.basename(x) for x in glob.glob(os.path.join(recdir, "*_annotated.json")) if
                      "paired_primary" not in x]

        for rec_sample_file in rec_tumors:
            prim_tumor_file = os.path.join(recdir, "paired_primary_tumor_" + rec_sample_file)
            rec_tumor_file = os.path.join(recdir, rec_sample_file)
            with open(prim_tumor_file) as f:
                prim_json = json.load(f)
            with open(rec_tumor_file) as f:
                rec_json = json.load(f)

            opt_w = optimize_weights(prim_json, rec_json)
            print("Optimized fitness component weights:")
            print(opt_w)

            sample = rec_json["id"]

#            print(fitness_weights[sample])
#            print("-----")

            cohort = prim_json["cohort"]

            predict(prim_json, opt_w) #fitness_weights[sample])

            # log-likelihood score
#            N_eff1 = effective_sample_size[sample]
            N_eff = rec_json["Effective_N"]
#            print(N_eff, N_eff1)
            prediction_KL = KL_prediction(prim_json, rec_json)
            prediction_ML = -N_eff * prediction_KL - 2 * np.log(N_eff) / 2
            null_KL = KL_null(prim_json, rec_json)
            null_ML = -N_eff * null_KL
            delta_ll = prediction_ML - null_ML
            aggregated_LL[cohort] += delta_ll
            lls[cohort].append(delta_ll)

    print("Number of samples with positive log-likelihood score:")
    npos_sts = sum([x > 0 for x in lls["STS"]])
    npos_lts = sum([x > 0 for x in lls["LTS"]])
    nsts = len([x > 0 for x in lls["STS"]])
    nlts = len([x > 0 for x in lls["LTS"]])
    print("STS: " + str(npos_sts) + "/" + str(nsts) + ", LTS: " + str(npos_lts) + "/" + str(nlts))
    print("Average log-likelihood score:")
    print("STS: " + str(np.mean([x for x in lls["STS"]])) + ", LTS: " + str(np.mean([x for x in lls["LTS"]])))
    print("Median log-likelihood score:")
    print("STS: " + str(np.median([x for x in lls["STS"]])) + ", LTS: " + str(np.median([x for x in lls["LTS"]])))
    print("Aggregated log-likelihood score:")
    print("STS: " + str(aggregated_LL["STS"]) + ", LTS: " + str(aggregated_LL["LTS"]))
