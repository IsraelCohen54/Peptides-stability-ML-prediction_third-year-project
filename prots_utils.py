from math import floor
import pandas as pd
import numpy as np
import torch

AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')
tail = False         # true to refer to stability of the peptide

# get gene ID from peptides, stable and not stable
def peptidesIds():
    pepsData = pd.read_excel("pepsData.xlsx")
    # extract ids column, transform to list
    ids_column = pepsData.pop("Gene_id")
    ids = ids_column.tolist()
    # remove peptides without gene_id
    ids = [ids[i] for i in range(len(ids)) if ids[i] != '0']
    # get highest and lowest quarters
    first_quarter_ids, last_quarter_ids = ids[:floor(len(ids) / 4)], ids[floor(len(ids) * 3 / 4):]
    return first_quarter_ids, last_quarter_ids


# shuffle two given list in the same way
def myShuffle(a, b):
    # make list of shuffled indices
    inds = np.arange(len(a))
    np.random.shuffle(inds)
    # make new arrays and give them shuffled values
    newA, newB = [], []
    for ind in inds:
        newA.append(a[ind])
        newB.append(b[ind])
    return newA, newB


# densely encode given seqs and labels, with last 1 or 0 for stable or unstable
def myEncode(prot_seqs, prot_ids):
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    def splitToChars(string):
        return [char for char in string]

    # dense-encode normally
    prots_seqs_encoded_ided = [(myEncodeDenseOne(np.array(splitToChars(seq))), id) for seq, id in zip(prot_seqs, prot_ids) if
                               isinstance(seq, str)]  # not sure why, but we get nan-float seqs. the 'if' filters them

    # get ids of stable and unstable peptides tails
    pep_UnstableID, pep_stableID = peptidesIds()
    # make lists of final prot seqs, with last num according to tail stability
    prot_seqs_final = []
    if tail:
        for prot_seq_encoded, prot_id in prots_seqs_encoded_ided:
            if prot_id in pep_stableID:
                prot_seqs_final.append(prot_seq_encoded + [1])
            elif prot_id in pep_UnstableID:
                prot_seqs_final.append(prot_seq_encoded + [-1])
            else:
                prot_seqs_final.append(prot_seq_encoded + [0])
    else:
        for prot_seq_encoded, prot_id in prots_seqs_encoded_ided:
            prot_seqs_final.append(prot_seq_encoded + [0])

    return prot_seqs_final


# make data of protein seqs
def makeInputData():
    protsData = pd.read_csv("protsData.csv")
    # extract important columns, transform to lists, take stable unstable samples (assume the csv comes sorted)
    prot_ids = protsData.pop("Gene_ID").tolist()
    prot_ids_unstable, prot_ids_stable = prot_ids[:floor(len(prot_ids) / 4)], prot_ids[floor(len(prot_ids) * 3 / 4):]
    prot_seqs = protsData.pop("AA_seq").tolist()
    prot_seqs_unstable, prot_seqs_stable = prot_seqs[:floor(len(prot_seqs) / 4)], prot_seqs[floor(len(prot_seqs) * 3 / 4):]
    # encode - with 1 for stable tail and -1 for unstable one, else 0
    prot_seqs_unstable_encoded = myEncode(prot_seqs_unstable, prot_ids_unstable)
    prot_seqs_stable_encoded = myEncode(prot_seqs_stable, prot_ids_stable)

    # unify and make labels, shuffle, transform to tensor
    seqs = prot_seqs_unstable_encoded + prot_seqs_stable_encoded
    labels = [0] * len(prot_seqs_unstable_encoded) + [1] * len(prot_seqs_stable_encoded)
    seqs, labels = myShuffle(seqs, labels)
    seqs, labels = torch.FloatTensor(seqs), torch.FloatTensor(labels)
    return seqs, labels



