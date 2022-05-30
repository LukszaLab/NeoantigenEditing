#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script for computing clone frequency predictions
    Copyright (C) 2022 Marta Luksza
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def set_fitness_on_a_tree(tree, weights):
    '''
    Sets fitness attribute to all clones, computed as the weighted sum of the immune and
    driver gene component.

    :param tree: dict
        json representation of a  tree

    :param weights: dict
        weights of the fitness model components, Sigma_I and Sigma_P.

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


#def compute_predicted_cumulative_frequencies_rec(node):
#    if "children" not in node:
#        node["predicted_X"] = node["predicted_x"]
#    else:
#        cvals = [compute_predicted_cumulative_frequencies_rec(child) for child in node["children"]]
#        node["predicted_X"] = sum(cvals)
#    return node["predicted_X"]


def compute_predicted_cumulative_frequencies(tree):
    '''
    Computes \hat X^\alpha_{rec}, Eq. 19 from Methods,
    the predicted inclusive clone frequency (CCF). Sets "predicted_X" attribute
    of all clones.
    '''
    def path2root(nid, parent):
        if nid == 0:
            return [nid]
        else:
            rpath = [nid] + path2root(parent[nid], parent)
            return rpath

    id2node = {}
    parent = {}
    nodes = [tree["topology"]]
    while nodes:
        node = nodes[0]
        node["predicted_X"] = 0
        id2node[node["clone_id"]] = node
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                parent[child["clone_id"]] = node["clone_id"]
                nodes.append(child)
    for nid in id2node:
        rpath = path2root(nid, parent)
        for aid in rpath:
            id2node[aid]["predicted_X"] += id2node[nid]["predicted_x"]


def predict_on_a_tree(tree):
    '''
    Compute "predicted_x" attribute of all clones (Eq. 18, Methods)
    '''
    nodes = [tree["topology"]]
    Z = 0.0
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        node["predicted_x"] = node["x"] * np.exp(node["fitness"])
        Z += node["predicted_x"]

    # normalize frequencies
    nodes = [tree["topology"]]
    while nodes:
        node = nodes[0]
        nodes = nodes[1:]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)
        node["predicted_x"] /= Z
    compute_predicted_cumulative_frequencies(tree)


def get_property(tree, property):
    '''
    Auxiliary function
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


def predict(tumor_json, weights):
    for tree in tumor_json["sample_trees"]:
        set_fitness_on_a_tree(tree, weights)
        predict_on_a_tree(tree)


def gather_clone_predictions(prim_json, rec_json):
    # only the top tree
    primary_tree = prim_json["sample_trees"][0]
    recurrent_tree = rec_json["sample_trees"][0]

    clone_ids = get_property(primary_tree, "clone_id")
    X_prim = get_property(primary_tree, "X")
    X_rec = get_property(recurrent_tree, "X")
    X_pred = get_property(primary_tree, "predicted_X")

    fit = get_property(primary_tree, "fitness")
    tab = pd.DataFrame(zip(X_prim, X_rec, X_pred, fit, clone_ids))
    tab.columns = ["X_prim", "X_rec", "X_pred", "Fitness", "Clone"]

    tab = tab[tab.X_prim > 0.03]
    tab = tab[[not (r.X_rec < 0.03 and r.X_prim < 3 * 0.03) for r in tab.itertuples()]]

    obs = [(r.X_rec) / (r.X_prim) for r in tab.itertuples()]
    mod = [(r.X_pred) / (r.X_prim) for r in tab.itertuples()]

    return obs, mod, list(tab.X_prim), list(tab.X_rec), list(tab.X_pred)


def plot_W_Wh(dobs, dmod, dX1, dX2, dcors):
    for name in ["LTS", "STS"]:
        lobs = [round(np.log2(x),6) for x in dobs[name]]
        lmod = [round(np.log2(x),6) for x in dmod[name]]
        cmap = plt.get_cmap('summer')
        fig, ax = plt.subplots(figsize=(9, 8))
        scatter = ax.scatter([max(0.08, x) for x in dobs[name]],
                             [max(0.08, x) for x in dmod[name]],
                             cmap=cmap,
                             s=(np.array(dX2[name]) * 2) ** (0.8) * 100,
                             c=np.sqrt(np.array(dX1[name])), alpha=0.99)

        kw = dict(prop="sizes", num=5,
                  func=lambda s: ((s / 100) ** (1 / 0.8)) / 2)
        legend2 = ax.legend(*scatter.legend_elements(**kw),
                            # loc="lower right",
                            title=r'$X_{\rm{rec}}$',
                            loc=(1.13, 0.01)
                            )
        ax.add_artist(legend2)
        kw = dict(prop="colors", num=5,
                  func=lambda s: s ** 2)
        ax.legend(*scatter.legend_elements(**kw),
                  title=r'$X_{\rm{prim}}$',
                  loc=(1.01, 0.01))

        ax.plot([0, 12], [0, 12], 'k--')
        ax.plot([0, 12], [1, 1], 'k--')
        ax.plot([1, 1], [0, 12], 'k--')
        ax.set_xlim(0.07, 12)
        ax.set_ylim(0.07, 12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Observed clone growth, " + r'$X_{\rm{rec}}/X_{\rm{prim}}$')
        ax.set_ylabel("Model clone fitness, " + r'$\hat X_{\rm{rec}}/X_{\rm{prim}}$')
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
        p_pval = int(np.floor(np.log10(dcors[name]["pearson_clones_pval"])) + 1)
        str_p_pval = 'p <=' + str(10 ** p_pval)
        s_pval = int(np.floor(np.log10(dcors[name]["spearman_clones_pval"])) + 1)
        str_s_pval = 'p <=' + str(10 ** s_pval)
        tit = name
        tit += "\npearson=" + str(round(dcors[name]["pearson_clones"], 2)) + "(" + str_p_pval + ")"
        tit += ", spearman=" + str(round(dcors[name]["spearman_clones"], 2)) + "(" + str_s_pval + ")"
        ax.set_title(tit)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.savefig(os.path.join("Results", "Figure_4b_" + name + ".pdf"), bbox_inches='tight')


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

    prim_observed_freq = {"LTS": [], "STS": []}
    rec_observed_freq = {"LTS": [], "STS": []}
    rec_predicted_freq = {"LTS": [], "STS": []}
    observed_freq_ratio = {"LTS": [], "STS": []}
    predicted_freq_ratio = {"LTS": [], "STS": []}

    dat = []
    for pat in patdirs:
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

            # clone frequency predictions
            obs, mod, X_prim, X_rec, X_pred = gather_clone_predictions(prim_json, rec_json)
            prim_observed_freq[cohort] += X_prim
            rec_observed_freq[cohort] += X_rec
            rec_predicted_freq[cohort] += X_pred
            observed_freq_ratio[cohort] += obs
            predicted_freq_ratio[cohort] += mod

    dcors = {}
    for i, coh in enumerate(["STS", "LTS"]):
        z = list(zip([round(x, 6) for x in observed_freq_ratio[coh]],
                     [round(x, 6) for x in predicted_freq_ratio[coh]]))
        obs1 = [el[0] for el in z]
        mod1 = [el[1] for el in z]
        c_pearson, c_pval = pearsonr(np.log(np.array(obs1)), np.log(np.array(mod1)))
        c_spearman, c_spval = spearmanr(obs1, mod1)
        dcors[coh] = {"pearson_clones": c_pearson, "pearson_clones_pval": c_pval,
                      "spearman_clones": c_spearman, "spearman_clones_pval": c_spval}

        dat = pd.DataFrame(list(zip(observed_freq_ratio[coh],
                                    predicted_freq_ratio[coh],
                                    prim_observed_freq[coh],
                                    rec_observed_freq[coh],
                                    rec_predicted_freq[coh],
                                    obs1, mod1
                                    )))
        dat.columns = ["Observed", "Model", "X_prim", "X_rec", "X_pred", "X_rec/X_prim", "X_pred/X_prim"]
        if not os.path.exists("Results"):
            os.mkdir("Results")

        dat.to_csv(os.path.join("Results", "Fig4b_" + coh + ".txt"), sep="\t", index=False)

    plot_W_Wh(observed_freq_ratio,
              predicted_freq_ratio,
              prim_observed_freq,
              rec_observed_freq, dcors)
