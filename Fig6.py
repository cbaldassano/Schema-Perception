from matplotlib import pyplot as plt
import numpy as np
import deepdish as dd
from util import nullZ, SRM
import brainiak.eventseg.event
from stimulus_annot import nStories, schema_type, design_intact, stories
import sys

nPerm = 1000
SRM_features = 100
ROI = sys.argv[1]
print('Running Fig 4 analysis for ' + ROI + '...')

story_start_TR = 8  # First TR after count-down stimulus
max_ev = 6  # Max number of events to try

# Load data
print('  Loading ' + ROI + '...')
native_D = dd.io.load('../data/Main/' + ROI + '.h5')
print('  Applying SRM...')
D = SRM(native_D, SRM_features)

np.random.seed(0)
acc = np.zeros((max_ev-1, 2, (8*8 - 8)//2, nPerm+1))
for s_i, s in enumerate(['R', 'A']):
    # Get ground truth event labels from design matrices
    designs = [M[story_start_TR:, :] for M in design_intact[schema_type == s]]
    gt_labels = np.empty(nStories//2, dtype='O')
    for i in range(nStories//2):
        gt_labels[i] = np.argmax(designs[i], axis=1)

    # Fit event models with varying numbers of events
    for n_ev in range(2, max_ev+1):
        print('  Fitting schema ' + s + ' with ' + str(n_ev) + ' events')
        ev_schema = brainiak.eventseg.event.EventSegment(n_ev)
        ev_schema.fit([D[k][:, story_start_TR:, :].mean(2).T
                       for k in np.array(stories)[schema_type == s]])

        # Get predicted labels (and null labels)
        pred_labels = np.empty(nStories//2, dtype='O')
        for i in range(nStories//2):
            pred_labels[i] = np.zeros((nPerm+1, designs[i].shape[0]))
            pred_labels[i][0, :] = np.argmax(ev_schema.segments_[i], axis=1)
            for p in range(nPerm):
                bounds = np.random.choice(pred_labels[i].shape[1],
                                          size=n_ev-1, replace=False)
                pred_labels[i][p+1, bounds] = 1
                pred_labels[i][p+1, :] = np.cumsum(pred_labels[i][p+1, :])

        # Compute between-story correspondence accuracy
        pair_i = 0
        for i in range(nStories//2):
            for j in range(i+1, nStories//2):
                gt_pair = gt_labels[i][:, np.newaxis] == \
                          gt_labels[j][np.newaxis, :]
                for p in range(nPerm+1):
                    pred_pair = pred_labels[i][p, :, np.newaxis] == \
                                pred_labels[j][p, np.newaxis, :]
                    acc[n_ev-2, s_i, pair_i, p] = \
                        (np.logical_and(gt_pair, pred_pair).sum() /
                         np.logical_or(gt_pair, pred_pair).sum())
                pair_i += 1

plt.figure()
plt.plot(np.arange(2, max_ev+1), nullZ(acc.mean(2).mean(1)))
plt.plot([2, 6], [2.876, 2.876], linestyle='--', color='black')
plt.xticks(np.arange(2, max_ev+1))
plt.ylabel('Match to annotations (z)')
plt.xlabel('Number of events')
plt.legend([ROI, 'p<0.01'])
plt.savefig('../results/Fig6_' + ROI + '.png')
print(' ')
