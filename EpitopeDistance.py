#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for computing the crossreactivity distance between two epitopes
    Copyright (C) 2022 Zachary Sethna

    Use is subject to the included term of use found at
    https://github.com/LukszaLab/NeoantigenEditing
"""
import numpy as np
import json
import os

#%
class EpitopeDistance(object):
    """Base class for epitope crossreactivity.

    Model:
        dist({a_i}, {b_i}) = \sum_i d_i M_ab(a_i, b_i)

    Attributes
    ----------
    amino_acids : str
        Allowed amino acids in specified order.

    amino_acid_dict : dict
        Dictionary of amino acids and corresponding indicies

    d_i : ndarray
        Position scaling array d_i.
        d_i.shape == (9,)

    M_ab : ndarray
        Amino acid substitution matrix. Indexed by the order of amino_acids.
        M_ab.shape == (20, 20)


    """

    def __init__(
        self,
        model_file=os.path.join(
            os.path.dirname(__file__), "data", "epitope_distance_model_parameters.json"
        ),
        amino_acids="ACDEFGHIKLMNPQRSTVWY",
    ):
        """Initialize class and compute M_ab."""

        self.amino_acids = amino_acids
        # self.amino_acid_dict = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.amino_acid_dict = {}
        for i, aa in enumerate(self.amino_acids):
            self.amino_acid_dict[aa.upper()] = i
            self.amino_acid_dict[aa.lower()] = i

        self.set_model(model_file)

    def set_model(self, model_file):
        """Load model and format substitution matrix M_ab."""
        with open(model_file, "r") as modelf:
            c_model = json.load(modelf)
        self.d_i = c_model["d_i"]
        self.M_ab_dict = c_model["M_ab"]
        M_ab = np.zeros((len(self.amino_acids), len(self.amino_acids)))
        for i, aaA in enumerate(self.amino_acids):
            for j, aaB in enumerate(self.amino_acids):
                M_ab[i, j] = self.M_ab_dict[aaA + "->" + aaB]
        self.M_ab = M_ab

    def epitope_dist(self, epiA, epiB):
        """Compute the model difference between the 9-mers epiA and epiB.

        Ignores capitalization.

        Model:
            dist({a_i}, {b_i}) = \sum_i d_i M_ab(a_i, b_i)
        """

        return sum(
            [
                self.d_i[i]
                * self.M_ab[
                    self.amino_acid_dict[epiA[i]], self.amino_acid_dict[epiB[i]]
                ]
                for i in range(9)
            ]
        )
