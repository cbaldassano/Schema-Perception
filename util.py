import numpy as np


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
