import numpy as np
import brainiak.funcalign.srm
from scipy.stats import zscore


# Compute Z score of real data relative to null distribution
def nullZ(X):
    # Last dimension of X is nPerm+1, with real data at 0 element
    X_roll = np.rollaxis(X, len(X.shape)-1)
    means = X_roll[1:].mean(axis=0)
    std = X_roll[1:].std(axis=0)
    if len(X.shape) > 1:
        std[std == 0] = np.nan
    Z = (X_roll[0] - means) / std
    return Z


# Permute items with the same label
def perm_groups(labels):
    groups = ''.join(set(labels))
    perm = np.zeros(len(labels), dtype=int)
    for g in groups:
        group_inds = np.where([g == x for x in labels])[0]
        perm[group_inds] = np.random.permutation(group_inds)

    return perm


# Run SRM on list of concatenated stories, then break into stories
def SRM_from_list(native_subj, story_breaks, story_names, nFeatures):
    srm = brainiak.funcalign.srm.SRM(features=nFeatures)
    srm.fit(native_subj)
    shared = srm.transform(native_subj)
    shared = zscore(np.dstack(shared), axis=1, ddof=1)

    shared_stories = dict()
    for i, k in enumerate(story_names):
        shared_stories[k] = shared[:, story_breaks[i]:story_breaks[i+1], :]
    return shared_stories


# Concatenate data into format expected by SRM
def data_list(D):
    nSubj = D[list(D.keys())[0]].shape[2]
    subj = []
    for s in range(nSubj):
        subj.append(np.concatenate([D[k][:, :, s] for k in D], axis=1))
        subj[s] = subj[s][np.logical_not(np.any(np.isnan(subj[s]), axis=1)), :]
    story_breaks = np.insert(0, 1, np.cumsum([D[k].shape[1] for k in D]))
    story_names = [k for k in D]
    return (subj, story_breaks, story_names)


# Align data across subjects into SRM feature space
def SRM(native_D, nFeatures):
    native_subj, story_breaks, story_names = data_list(native_D)
    D = SRM_from_list(native_subj, story_breaks, story_names, nFeatures)
    return D
