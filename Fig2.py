import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from scipy.stats import zscore, norm
from util import nullZ, SRM
from deconvolve import deconv
from stimulus_annot import modality, mask, design_intact, stories, nStories
import sys
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

ROI = sys.argv[1]

# All SRM dimensions for Fig 2-1, and dimensions for Fig 2
SRM_features = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]
SRM_features_Fig2 = 100
nPerm = 1000

print('Running Fig 2 and 2-1 analysis for ' + ROI + '...')

# Load data
print('  Loading ' + ROI + '...')
native_D = dd.io.load('../data/Main/' + ROI + '.h5')
nSubj = native_D[stories[0]].shape[2]

schema_z = np.zeros(len(SRM_features))

for dim_i, dim in enumerate(SRM_features):
    print('  Computing correlations with with ' + str(dim) + ' SRM dimensions')
    D = SRM(native_D, dim)

    print('  Computing correlations...')
    # Compute event correlations between random splits of data
    nSplit = 10
    story_corr = np.zeros((nStories, nStories, nSplit))
    np.random.seed(0)
    for p in range(nSplit):
        group1_beta = np.zeros((nStories, dim, 4))
        group2_beta = np.zeros((nStories, dim, 4))

        # Deconvolve event representations
        group_perm = np.random.permutation(nSubj)
        for i in range(nStories):
            G1 = np.nanmean(D[stories[i]][:, :, group_perm[:16]], axis=2)
            G2 = np.nanmean(D[stories[i]][:, :, group_perm[16:]], axis=2)
            group1_beta[i, :, :] = zscore(deconv(G1, design_intact[i])[:, 1:],
                                          axis=0, ddof=1)
            group2_beta[i, :, :] = zscore(deconv(G2, design_intact[i])[:, 1:],
                                          axis=0, ddof=1)

        # Correlate each event between all stories
        for ev in range(4):
            story_corr[:, :, p] += (1/4) * \
                np.dot(group1_beta[:, :, ev], group2_beta[:, :, ev].T)/(dim-1)

    # Collapse across splits and symmetrize
    story_corr = story_corr.mean(2)
    story_corr = (story_corr + story_corr.T)/2

    # Compute true and null similarity for all pairs
    within_schema = np.zeros(nPerm+1)
    across_schema = np.zeros(nPerm+1)
    np.random.seed(0)
    SC_perm = story_corr
    for p in range(nPerm+1):
        within_schema[p] = np.mean([np.mean(SC_perm[mask == 2]),
                                    np.mean(SC_perm[mask == 3])])
        across_schema[p] = np.mean([np.mean(SC_perm[mask == 4]),
                                    np.mean(SC_perm[mask == 5])])

        nextperm = np.random.permutation(story_corr.shape[0])
        SC_perm = story_corr[np.ix_(nextperm, nextperm)]

    schema_z[dim_i] = nullZ(within_schema - across_schema)

    # Only output full results for one SRM dimension
    if dim != SRM_features_Fig2:
        continue

    # Compute true and null similarity for cross-modal pairs
    within_schema_crossmod = np.zeros(nPerm+1)
    across_schema_crossmod = np.zeros(nPerm+1)
    np.random.seed(0)
    SC_perm = story_corr
    for p in range(nPerm+1):
        within_schema_crossmod[p] = np.mean(SC_perm[mask == 3])
        across_schema_crossmod[p] = np.mean(SC_perm[mask == 5])

        # Permute only within modalities
        groups = ''.join(set(modality))
        nextperm = np.zeros(len(modality), dtype=int)
        for g in groups:
            group_inds = np.where([g == x for x in modality])[0]
            nextperm[group_inds] = np.random.permutation(group_inds)
        SC_perm = story_corr[np.ix_(nextperm, nextperm)]

    print('  ' + ROI + ' all pairs p=' +
          str(norm.sf(nullZ(within_schema - across_schema))))
    print('  ' + ROI + ' cross-mod p=' +
          str(norm.sf(nullZ(within_schema_crossmod - across_schema_crossmod))))

    # Plot Fig 2 results and null distributions
    plt.figure()
    schema_diff = within_schema - across_schema
    crossmod_diff = within_schema_crossmod - across_schema_crossmod
    plt.plot(0, schema_diff[0], marker='o', markersize=6)
    plt.plot(1, crossmod_diff[0], marker='o', markersize=6)
    vp = plt.violinplot(schema_diff[1:], positions=[0], showextrema=False)
    vp['bodies'][0].set_color('0.8')
    vp = plt.violinplot(crossmod_diff[1:], positions=[1], showextrema=False)
    vp['bodies'][0].set_color('0.8')
    plt.ylabel('Event correlation W vs A (r)')
    plt.ylim([0, 0.15])
    plt.xticks([0, 1], ['All pairs', 'Cross-mod'])
    plt.savefig('../results/Fig2_' + ROI + '.png')

# Plot Fig 2-1 curve
plt.figure(figsize=(10, 6))
z_smooth = lowess(schema_z, SRM_features)
plt.plot(z_smooth[:, 0], z_smooth[:, 1])
plt.plot(SRM_features, np.ones(len(SRM_features))*norm.isf(0.05), '--')
plt.plot(SRM_features, np.ones(len(SRM_features))*norm.isf(0.01), '--')
plt.gca().set_xscale('log')
ticks = [2, 5, 10, 20, 50, 100]
plt.xticks(ticks, [str(x) for x in ticks])
plt.minorticks_off()
plt.xlabel('SRM Dimensions')
plt.ylabel('Schema effect (z)')
plt.savefig('../results/Fig2-1_' + ROI + '.png')

print(' ')
