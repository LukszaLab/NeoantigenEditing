#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script for computing alignments of neoantigens to IEDB peptides
    Copyright (C) 2022 Marta Luksza
"""


import glob
import json
import os
import subprocess
import tempfile
from collections import defaultdict
import numpy as np

import pandas as pd
from Bio import SeqIO
from Bio.pairwise2 import align

def load_blosum62_mat():
    raw_blosum62_mat_str = '''
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4 
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4 
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4 
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4 
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4 
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4 
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4 
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4 
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4 
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4 
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4 
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4 
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4 
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4 
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4 
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4 
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4 
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4 
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4 
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4 
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4 
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4 
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
'''
    amino_acids='ACDEFGHIKLMNPQRSTVWY'
    blosum62_mat_str_list = [l.split() for l in raw_blosum62_mat_str.strip().split('\n')]
    blosum_aa_order = [blosum62_mat_str_list[0].index(aa) for aa in amino_acids]

    blosum62_mat = np.zeros((len(amino_acids), len(amino_acids)))
    for i, bl_ind in enumerate(blosum_aa_order):
        blosum62_mat[i] = np.array([int(x) for x in blosum62_mat_str_list[bl_ind + 1][1:]])[blosum_aa_order]
    blosum62 = {(aaA, aaB): blosum62_mat[i, j] for i, aaA in enumerate(amino_acids)
                         for j, aaB in enumerate(amino_acids)}
    return blosum62


def align_peptides(seq1, seq2, matrix):
    gap_open = -11
    gap_extend = -1
    aln = align.localds(seq1.upper(), seq2.upper(), matrix, gap_open, gap_extend)
    return aln[0]


def run_blastp_n(pep_list, blastdb):
    '''
    Run BLASTP on the given n neoantigens

    :param pep_list: list
        list of peptides (neoantigen sequnces)

    :param blastdb: str
        fasta file with IEDB peptides

    :return: dict
        str (peptide) -> list of IEDB identifiers
    '''

    if blastdb is None:
        raise ValueError("No BLAST database specified")
    os_fid, fa_file = tempfile.mkstemp(suffix=".fa", dir=os.getcwd())
    os.close(os_fid)
    os_fid, txt_file = tempfile.mkstemp(suffix=".txt", dir=os.getcwd())
    os.close(os_fid)
    id2seq = {}
    seq2id = {}
    with open(fa_file, "w") as fh:
        for seqid, neoseq in enumerate(pep_list):
            seqid2 = "seq_" + str(seqid)
            # seqid2id[seqid2] = seqid
            id2seq[seqid2] = neoseq
            seq2id[neoseq] = seqid2
            fh.write(">seq_{}\n{}\n".format(seqid, neoseq))
    # run BLASTP
    blastpexe = "blastp"
    blast_args = [blastpexe, "-db", blastdb, "-query", fa_file,
                  "-outfmt", "6 qseqid sacc score",
                  "-gapopen", "32767", "-gapextend", "32767",
                  "-evalue", "1e6", "-max_hsps", "1", "-matrix", "BLOSUM62",
                  "-max_target_seqs", "10000000", "-out", txt_file]

    subprocess.check_call(blast_args)
    os.unlink(fa_file)
    alignments = defaultdict(list)
    with open(txt_file) as fh:
        for line in fh:
            S = line.split()
            epi_id = int(S[1].split("|")[0])
            seq_id = id2seq[S[0]]
            alignments[seq_id].append(epi_id)
    os.unlink(txt_file)
    return alignments


def run_blastp(peplist, blastdb, n=1000):
    '''
    Blast peptides in neolist against peptides in blastdb.

    :param peplist: list
        list of peptides (neoantigens)

    :param blastdb:
        iedb fasta file

    :param n: int
        run blastp in batches of size n

    :return: dict
        dictionary mapping neoantigen peptide sequences to alignment candidates
    '''

    alignments = defaultdict(set)
    for i in range(0, len(peplist) + n, n):  # run blastp in batches of size n
        peplist0 = peplist[i:(i + n)]
        if len(peplist0) == 0:
            continue
        alignments0 = run_blastp_n(peplist0, blastdb)
        for pepseq in alignments0:
            for epi in alignments0[pepseq]:
                alignments[pepseq].add(epi)
    return alignments


def prepare_blastdb(peptidesfasta):
    '''
    Builds BLAST database

    :param peptidesfasta: str
        path to the IEDB.fasta file
    '''
    instr = ["makeblastdb", "-in", peptidesfasta, "-dbtype", "prot", ">", "/dev/null"]
    instr = "\t".join(instr)
    os.system(instr)


def load_epitopes(iedbfasta):
    '''
    Load IEDB epitopes from fasta file

    :param iedbfasta: str

    :return: dict
        IEDB epitope identifiers mapped to epitope sequence
    '''
    epitopes = {}
    with open(iedbfasta) as f:
        seqs = SeqIO.parse(f, "fasta")
        for seq in seqs:
            seqid = int((seq.id).split("|")[0])
            epitopes[seqid] = str(seq.seq)
    return epitopes


if __name__ == "__main__":

    '''
    
    Aligns neoantigens peptides of all patients to IEDB
    Requirement: blastp installed and in the PATH
    
    run as:
    python align_neoantigens_to_IEDB.py
     
    '''

    dir = os.path.join("data")
    patient_dir = os.path.join("data", "Patient_data")
    iedb_file = os.path.join("data", "iedb.fasta")

    if not os.path.exists(os.path.join("data", "IEDB_alignments")):
        os.mkdir(os.path.join("data", "IEDB_alignments"))
    # blosum62

    blosum62 = load_blosum62_mat()

    # prepare blast database
    prepare_blastdb(iedb_file)
    epitopes = load_epitopes(iedb_file)

    patientdirs = glob.glob(os.path.join(patient_dir, "*", "Primary"))
    # file to extract neoantigen sequences for that patient
    patientfiles = [glob.glob(os.path.join(pdir, "*.json"))[0] for pdir in patientdirs]

    for pfile in patientfiles:
        with open(pfile) as f:
            pjson = json.load(f)
        patient = pjson["patient"]
        neoantigens = pjson["neoantigens"]
        peptides = set([("_".join(neo["id"].split("_")[:-1]), neo["sequence"]) for neo in neoantigens])
        pepseq2pepid = defaultdict(set)
        for pep_id, pep_seq in peptides:
            pepseq2pepid[pep_seq].add(pep_id)

        seqlist = list(set([pep_seq for pep_id, pep_seq in peptides]))
        alignments = run_blastp(seqlist, iedb_file, n=100)
        scores = []
        aln_data = []
        for pep_seq in alignments:
            for epitope_id in alignments[pep_seq]:
                episeq = epitopes[epitope_id]
                score = align_peptides(pep_seq, episeq, blosum62).score
                pep_ids = pepseq2pepid[pep_seq]
                for pep_id in pep_ids:
                    aln_data.append([pep_id, pep_seq, epitope_id, score])
        if len(aln_data):
            aln_data = pd.DataFrame(aln_data)
            aln_data.columns = ["Peptide_ID", "Peptide", "Epitope_ID", "Alignment_score"]
            aln_data.to_csv(os.path.join("data", "IEDB_alignments", "iedb_alignments_" + patient + ".txt"), sep="\t",
                            index=False)
