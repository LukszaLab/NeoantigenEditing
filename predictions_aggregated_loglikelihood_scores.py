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


def predict_on_a_tree(tree):
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


def get_property(tree, property):
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
    x0 = get_property(prim_tree, "tilde_x")
    observed_x = get_property(rec_tree, "tilde_x")
    pairs = list(zip(x0, observed_x))
    pairs = [x for x in pairs if x[1] > 0 and x[0] > 0]

    kl = sum(obs_x * np.log(obs_x / null_x) for (null_x, obs_x) in pairs)
    return kl


def KL_prediction_tree(prim_tree, rec_tree):
    x0 = get_property(prim_tree, "tilde_x")
    observed_x = get_property(rec_tree, "tilde_x")
    predicted_x = get_property(prim_tree, "predicted_x")
    pairs = list(zip(predicted_x, observed_x, x0))
    pairs = [x for x in pairs if x[2] > 0 and x[1] > 0]

    kl = sum(obs_x * np.log(obs_x / pred_x) for (pred_x, obs_x, _) in pairs)
    return kl


def KL_prediction(prim_json, rec_json):
    prim_trees = prim_json["sample_trees"]
    rec_trees = rec_json["sample_trees"]
    weights = np.array([tree["score"] for tree in prim_json["sample_trees"]])
    weights = np.exp(weights - max(weights))
    weights = weights / sum(weights)
    kl = sum([w * KL_prediction_tree(prim_tree, rec_tree) for (prim_tree, rec_tree, w) in
              zip(prim_trees, rec_trees, weights)])
    return kl


def KL_null(prim_json, rec_json):
    prim_trees = prim_json["sample_trees"]
    rec_trees = rec_json["sample_trees"]
    weights = np.array([tree["score"] for tree in prim_json["sample_trees"]])
    weights = np.exp(weights - max(weights))
    weights = weights / sum(weights)
    kl = sum(
        [w * KL_null_tree(prim_tree, rec_tree) for (prim_tree, rec_tree, w) in zip(prim_trees, rec_trees, weights)])
    return kl


def predict(tumor_json, weights):
    for tree in tumor_json["sample_trees"]:
        set_fitness_on_a_tree(tree, weights)
        predict_on_a_tree(tree)


if __name__ == "__main__":

    '''
    Performs clone frequency predictions
     
    '''

    dir = os.path.join("data")
    patient_dir = os.path.join("data", "Patient_data")
    weights = pd.read_csv(os.path.join(dir, "fitness_weights.txt"), sep="\t")
    fitness_weights = {}
    effective_sample_size = {}
    for r in weights.itertuples():
        fitness_weights[r.Rec_sample_ID] = {"Sigma_I": r.Sigma_I, "Sigma_P": r.Sigma_P}
        effective_sample_size[r.Rec_sample_ID] = r.Effective_N
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
            cohort = prim_json["cohort"]
            sample = rec_json["id"]

            predict(prim_json, fitness_weights[sample])

            # log-likelihood score
            N_eff = effective_sample_size[sample]
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
