import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from scipy.stats import zscore, norm
from util import nullZ, SRM
from deconvolve import deconv
from stimulus_annot import modality, mask, design, stories, nStories
import sys

nPerm = 1000
SRM_features = 100
ROI = sys.argv[1]
print('Running Fig 2 analysis for ' + ROI + '...')

# Load data
print('  Loading ' + ROI + '...')
native_D = dd.io.load('../data/' + ROI + '.h5')
print('  Applying SRM...')
D = SRM(native_D, SRM_features)

dim = D[stories[0]].shape[0]
nSubj = D[stories[0]].shape[2]

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
        group1_mean = np.nanmean(D[stories[i]][:, :, group_perm[:16]], axis=2)
        group2_mean = np.nanmean(D[stories[i]][:, :, group_perm[16:]], axis=2)
        group1_beta[i, :, :] = zscore(deconv(group1_mean, design[i])[:, 1:],
                                      axis=0, ddof=1)
        group2_beta[i, :, :] = zscore(deconv(group2_mean, design[i])[:, 1:],
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
print('  ' + ROI + ' all pairs p=' +
      str(norm.sf(nullZ(within_schema - across_schema))))

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
print('  ' + ROI + ' cross-mod p=' +
      str(norm.sf(nullZ(within_schema_crossmod - across_schema_crossmod))))

# Plot results and null distributions
plt.figure()
plt.plot(0, within_schema[0] - across_schema[0],
         marker='o', markersize=6)
plt.plot(1, within_schema_crossmod[0] - across_schema_crossmod[0],
         marker='o', markersize=6)
vp = plt.violinplot(within_schema[1:] - across_schema[1:],
                    positions=[0], showextrema=False)
vp['bodies'][0].set_color('0.8')
vp = plt.violinplot(within_schema_crossmod[1:] - across_schema_crossmod[1:],
                    positions=[1], showextrema=False)
vp['bodies'][0].set_color('0.8')
plt.ylabel('Event correlation W vs A (r)')
plt.ylim([0, 0.15])
plt.xticks([0, 1], ['All pairs', 'Cross-mod'])
plt.savefig('../results/Fig2_' + ROI + '.png')

print(' ')
