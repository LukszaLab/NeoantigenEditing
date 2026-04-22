#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script for computing neoantigen qualities and fitness of tumor clones
    Copyright (C) 2022 Marta Luksza
"""

import argparse
import json

from collections import defaultdict

import numpy as np
import pandas as pd

from EpitopeDistance import EpitopeDistance


def fill_up_clone_mutations(tree, mut2missense):
    """
    Fills up the field with all mutations for each clone on the tree

    :param tree: dict
        imported from json file

    :param mut2missense: dict
        maps mutation identifiers to 1 if missense else to 0

    :return: dict
        json dictionary with filled up mutations
    """

    nodes = [(tree["topology"], [])]
    while len(nodes) > 0:
        (node, anc_mutations) = nodes[0]
        nodes = nodes[1:]
        cmutations = node["clone_mutations"]
        node["all_mutations"] = list(set(cmutations + anc_mutations))
        node["TMB"] = sum([mut2missense[mid] for mid in node["all_mutations"]])
        if "children" in node:
            for child in node["children"]:
                nodes.append((child, node["all_mutations"]))


def fill_up_clone_neoantigens(tree, mut2neo):
    """
    Adds neoantigen field for each clone on the tree

    :param tree: dict
        tree imported from json file

    :param mut2neo: dict
        str -> list
        mapts mutation identifiers to neoantigen entries

    :return: dict
        annotated json dictionary
    """

    nodes = [tree["topology"]]
    while len(nodes) > 0:
        node = nodes[0]
        nodes = nodes[1:]
        node["neoantigens"] = [
            neo["id"] for mid in node["all_mutations"] for neo in mut2neo[mid]
        ]
        node["neoantigen_load"] = len(node["neoantigens"])
        node["NA_Mut"] = sum([len(mut2neo[mid]) > 0 for mid in node["all_mutations"]])
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)


def mark_driver_gene_mutations(pjson):
    """
    Create a dictionary mapping mutation identifiers to their driver gene status
    (1 - in a driver gene, 0 - not in a driver gene)

    :param pjson: dict
        json representation of a sample

    :return dict
        (str -> int)
    """

    dg_genes = set(["TP53", "KRAS", "CDKN2A", "SMAD4"])
    mutid2dg = {}
    for mut in pjson["mutations"]:
        mutid2dg[mut["id"]] = mut["gene"] in dg_genes
    return mutid2dg


def mark_missense_mutations(pjson):
    """
    Create a dictionary mapping mutation identifiers to their missense status
    (1 - is a missense mutation, 0 - not a missense mutation)

    :param pjson: dict
        json representation of a sample

    :return dict
        (str -> int)
    """

    mut2missense = {}
    for mut in pjson["mutations"]:
        mut2missense[mut["id"]] = mut["missense"]
    return mut2missense


def map_neoantigen_qualities(pjson):
    neoid2quality = {}
    for neo in pjson["neoantigens"]:
        neoid2quality[neo["id"]] = neo["quality"]
    return neoid2quality


def get_property(tree, property):
    """
    Auxiliary function to extract a clone attribute values into a list
    :param tree: dict
        json representation of a tree

    :param property: str
        name of the attribute

    :return list
    """

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


def compute_effective_sample_size(sample_json):
    """
    Computes the effective cancer cell population size for each sample (see Methods, p.10)

    :param sample_json: dict
        json representation of a sample

    :return float
    """

    mut_freqs = {}
    for mut in sample_json["mutations"]:
        mut_freqs[mut["id"]] = []
    for tree in sample_json["sample_trees"]:
        clone_muts_list = get_property(tree, "clone_mutations")
        freqs = get_property(tree, "X")
        for clone_muts, X in zip(clone_muts_list, freqs):
            for mid in clone_muts:
                mut_freqs[mid].append(X)
    avev = np.mean(
        [np.var(mut_freqs[mid]) if mut_freqs[mid] else 0 for mid in mut_freqs]
    )
    n = 1 / avev
    return n


INF = float("inf")


def log_sum2(v1, v2):
    ma = max(v1, v2)
    if ma == -INF:
        return -INF
    return ma + np.log(np.exp(v1 - ma) + np.exp(v2 - ma))


def log_sum(v):
    if len(v):
        ma = max(v)
        if ma == -INF:
            return -INF
        return np.log(sum(list(map(lambda x: np.exp(x - ma), v)))) + ma
    return -INF


def compute_R(scores, a, k):
    """
    Computes the value of the R component given the alignment scores and parameters a and k

    :param  scores: list
        list of alignment scores for a given neoantigen and IEDB epitopes

    :param a: float
        shift parameter of the sigmoid function

    :param k: float
        slope parameter of the sigmoid function

    :return float

    """
    v = [-k * (a - score) for score in scores]
    lgb = log_sum(v)
    lZ = log_sum2(0, lgb)
    bindProb = np.exp(lgb - lZ)
    return bindProb


def set_immune_fitness(tree, neo2qualities):
    """
    Sets the value of the immune fitness component for each clone in the tree
    as the negative of the max quality neoantigen.

    :param tree: dict
        json representation of a tree

    :param neo2qualities: dict
        mapping neoantigen identifiers to their qualities (str->float)
    """

    nodes = [tree["topology"]]
    while len(nodes) > 0:
        node = nodes[0]
        nodes = nodes[1:]
        if node["neoantigens"]:
            node["F_I"] = -max([neo2qualities[neoid] for neoid in node["neoantigens"]])
        else:
            node["F_I"] = 0
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)


def set_driver_gene_fitness(tree, mut2dg):
    """
    Sets the driver-gene fitness component for each clone in the tree
    as the number of driver gene mutations in that clone.

    :param tree: dict
        json representation of a tree

    :param mut2dg: dict
        mapping mutation identifiers to 1 if this is a mutation in a driver gene or to 0.
    """
    nodes = [tree["topology"]]
    while len(nodes) > 0:
        node = nodes[0]
        nodes = nodes[1:]
        if node["all_mutations"]:
            node["F_P"] = sum([mut2dg[mut_id] for mut_id in node["all_mutations"]])
        else:
            node["F_P"] = 0
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)


def clean_data(tree):
    """
    Removes no longer needed clone attributes.
    """
    nodes = [tree["topology"]]
    while len(nodes) > 0:
        node = nodes[0]
        nodes = nodes[1:]
        del node["all_mutations"]
        del node["clone_mutations"]
        del node["neoantigens"]
        if "children" in node:
            for child in node["children"]:
                nodes.append(child)


if __name__ == "__main__":

    """
    Computes components contributing to neoantigen quality score, fitness of clones and annotates clones with neoantigens in *_annotated.json
    files.

    Run as:

    python compute_fitness.py

    """

    a = 22.897590714815188
    k = 1
    w = 0.22402192838740312

    parser = argparse.ArgumentParser(prog="align_neoantigens_to_IEDB")
    parser.add_argument("--alignment", help="neoantigen alignment file", required=True)
    parser.add_argument("--input", help="patient_data file", required=True)

    args = parser.parse_args()

    alignment_file = args.alignment
    patient_file = args.input

    epidist = EpitopeDistance()

    sample_file = patient_file
    output_file = patient_file.replace(".json", "_annotated.json")

    norm = 1

    with open(sample_file) as f:
        sjson = json.load(f)
    patient = sjson["patient"]
    neoantigens = sjson["neoantigens"]
    nalist = [neo["sequence"] for neo in neoantigens]

    alignments = pd.read_csv(alignment_file, sep="\t")
    naseq2scores = defaultdict(list)
    for r in alignments.itertuples():
        naseq2scores[r.Peptide].append(r.Alignment_score)

    mut2neo = defaultdict(list)
    for neo in neoantigens:
        score_list = naseq2scores[neo["sequence"]]
        neo["R"] = compute_R(score_list, a, k)
        neo["logC"] = epidist.epitope_dist(neo["sequence"], neo["WT_sequence"])
        neo["logA"] = np.log(neo["KdWT"] / neo["Kd"])
        neo["quality"] = (w * neo["logC"] + (1 - w) * neo["logA"]) * neo["R"]
        mut2neo[neo["mutation_id"]].append(neo)

    mut2dg = mark_driver_gene_mutations(sjson)
    mut2missense = mark_missense_mutations(sjson)
    neo2qualities = map_neoantigen_qualities(sjson)

    for tree in sjson["sample_trees"]:
        fill_up_clone_mutations(tree, mut2missense)
        fill_up_clone_neoantigens(tree, mut2neo)
        set_immune_fitness(tree, neo2qualities)
        set_driver_gene_fitness(tree, mut2dg)

    neff = compute_effective_sample_size(sjson)
    sjson["Effective_N"] = neff / norm

    for tree in sjson["sample_trees"]:
        clean_data(tree)

    with open(output_file, "w") as of:
        json.dump(sjson, of, indent=True)
