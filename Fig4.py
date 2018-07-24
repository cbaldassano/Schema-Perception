import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from scipy.stats import zscore, norm
from util import nullZ
from deconvolve import deconv
from stimulus_annot import mask, design_intact, design_scrambled, \
    design_scrambled_event, design_scrambled_story, stories, nStories, clips
import sys

ROI = sys.argv[1]
nPerm = 1000

print('Running Fig 4 analysis for ' + ROI + '...')

# Load data
print('  Loading ' + ROI + '...')
D_scrambled = dd.io.load('../data/Control/' + ROI + '.h5')
D_intact = dd.io.load('../data/Main/' + ROI + '.h5')

dim = D_intact[stories[0]].shape[0]
nSubj_intact = D_intact[stories[0]].shape[2]
nSubj_scrambled = D_scrambled[clips[0]].shape[2]

# Regression coefficients for scrambled stimuli
S_beta = np.zeros((nStories, dim, 4))
for i in range(len(clips)):
    S_mean = np.nanmean(D_scrambled[clips[i]], axis=2)
    beta = zscore(deconv(S_mean, design_scrambled[i]), axis=0, ddof=1)
    S_beta[design_scrambled_story[i], :, design_scrambled_event[i]] = beta.T

# Compute event correlations between random splits of intact with scrambled
nSplit = 10
story_corr_II = np.zeros((nStories, nStories, nSplit))
story_corr_IS = np.zeros((nStories, nStories, nSplit))
np.random.seed(0)
for p in range(nSplit):
    I1_beta = np.zeros((nStories, dim, 4))
    I2_beta = np.zeros((nStories, dim, 4))

    # Deconvolve event representations
    group_perm = np.random.permutation(nSubj_intact)
    for i in range(nStories):
        D_story = D_intact[stories[i]]
        I1 = D_story[:, :, group_perm[:nSubj_scrambled]]
        I2 = D_story[:, :, group_perm[nSubj_scrambled:(2*nSubj_scrambled)]]
        I1_mean = np.nanmean(I1, axis=2)
        I2_mean = np.nanmean(I2, axis=2)
        I1_beta[i, :, :] = zscore(deconv(I1_mean, design_intact[i])[:, 1:],
                                  axis=0, ddof=1)
        I2_beta[i, :, :] = zscore(deconv(I2_mean, design_intact[i])[:, 1:],
                                  axis=0, ddof=1)

    # Correlate each event between all stories
    for ev in range(4):
        story_corr_II[:, :, p] += (1/4) * \
            np.dot(I1_beta[:, :, ev], I2_beta[:, :, ev].T)/(dim-1)
        story_corr_IS[:, :, p] += (1/4) * \
            np.dot(I1_beta[:, :, ev], S_beta[:, :, ev].T)/(dim-1)


# Collapse across splits and symmetrize
story_corr_II = story_corr_II.mean(2)
story_corr_IS = story_corr_IS.mean(2)
story_corr_II = (story_corr_II + story_corr_II.T)/2
story_corr_IS = (story_corr_IS + story_corr_IS.T)/2


# Compute true and null similarity for all pairs
WvA_diff = np.zeros(nPerm+1)
np.random.seed(0)
SCII_p = story_corr_II.copy()
SCIS_p = story_corr_IS.copy()
for p in range(nPerm+1):
    WvA_II = np.mean([SCII_p[mask == 2].mean(), SCII_p[mask == 3].mean()]) - \
             np.mean([SCII_p[mask == 4].mean(), SCII_p[mask == 5].mean()])
    WvA_IS = np.mean([SCIS_p[mask == 2].mean(), SCIS_p[mask == 3].mean()]) - \
             np.mean([SCIS_p[mask == 4].mean(), SCIS_p[mask == 5].mean()])

    WvA_diff[p] = WvA_II - WvA_IS

    nextperm = np.random.permutation(story_corr_II.shape[0])
    SCII_p = story_corr_II[np.ix_(nextperm, nextperm)]
    SCIS_p = story_corr_IS[np.ix_(nextperm, nextperm)]

print('  WvA_diff:' + str(norm.sf(nullZ(WvA_diff))))

# Plot Fig 4 results and null distribution
plt.figure()
plt.plot(0, WvA_diff[0], marker='o', markersize=6)
vp = plt.violinplot(WvA_diff[1:], positions=[0], showextrema=False)
vp['bodies'][0].set_color('0.8')
plt.ylabel('Schema effect intact minus scrambled (r)')
plt.xticks([0], [ROI])
plt.savefig('../results/Fig4_' + ROI + '.png')

print(' ')
