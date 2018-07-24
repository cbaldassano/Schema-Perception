import numpy as np
from deconvolve import deconv
import deepdish as dd
import brainiak.eventseg.event
from scipy.stats import zscore, norm
from stimulus_annot import stories, schema_type, design_intact, story_run
from util import data_list, SRM_from_list
from matplotlib import pyplot as plt
import sys

ROI = sys.argv[1]
nBoot = int(sys.argv[2])
SRM_features = 100
print('Running Fig 5 and 5-1 analysis for ' + ROI + '...')

# Load data
print('  Loading ' + ROI + '...')
native_D = dd.io.load('../data/Main/' + ROI + '.h5')
native_subj, story_breaks, story_names = data_list(native_D)
nSubj = len(native_subj)

# Accuracies for both cross-validation types
acc = np.zeros((nBoot, 2))
np.random.seed(0)
for boot in range(nBoot):
    print('  Bootstrap ' + str(boot))
    boot_samp = np.random.randint(0, nSubj, nSubj)
    native_boot = [native_subj[i] for i in boot_samp]
    print('    Runing SRM...')
    D = SRM_from_list(native_boot, story_breaks, story_names, SRM_features)
    dim = D[stories[0]].shape[0]

    print('    Leaving 2 stories out...')
    # Fit an event model to 7 stories (all but one story from a schema),
    # then try to find these events in all 9 other stories
    evcorr = np.empty((16, 16))
    evcorr[:] = np.nan
    # Loop over choice of held-out story
    for loo in range(len(stories)):
        # Deconvolve event representations from 7 stories
        same_schema = (schema_type == schema_type[loo])
        same_schema[loo] = False
        all_design = np.concatenate(design_intact[same_schema], axis=0)
        all_D = np.empty((dim, 0))
        for i in np.where(same_schema)[0]:
            all_D = np.append(all_D, D[stories[i]].mean(2), axis=1)
        train_beta = zscore(deconv(all_D, all_design), axis=0, ddof=1)
        nEvents = train_beta.shape[1]

        # Create event model
        ev = brainiak.eventseg.event.EventSegment(nEvents)
        ev.set_event_patterns(train_beta)
        ev_var = ev.calc_weighted_event_var(all_D.T, all_design, train_beta)

        # Measure fit of event model to the 9 held-out stories
        diff_schema = np.where(schema_type != schema_type[loo])[0]
        for j in np.append(loo, diff_schema):
            pred_labels = ev.find_events(D[stories[j]].mean(2).T, ev_var)[0]
            pred_mean = np.matmul(D[stories[j]].mean(2), pred_labels)
            cc = np.corrcoef(train_beta.T, pred_mean.T)[:nEvents, nEvents:]
            d = np.diag(cc).sum()
            # Mean of diagonal minus mean of off-diagonal
            evcorr[loo, j] = d/nEvents - (cc.sum()-d)/(nEvents**2 - nEvents)

    # For pairs of held-out stories (one from each schema),
    # check whether they are correctly classified
    for lto_R in range(8):
        for lto_A in range(8, 16):
            corr_lto = evcorr[np.ix_([lto_R, lto_A], [lto_R, lto_A])]
            if (corr_lto[0, 0] + corr_lto[1, 1]) > \
               (corr_lto[0, 1] + corr_lto[1, 0]):
                acc[boot, 0] += 1/(8*8)

    print('    Leaving one run out...')
    # Fit an event model for each schena to 3 runs,
    # then try to find these events in the 4 remaining stories

    # Event correlations for each schema model
    evcorr = dict()
    for schema in np.unique(schema_type):
        evcorr[schema] = np.zeros(len(stories))

    # Loop over choice of held-out run
    for loro in np.unique(story_run):
        for schema in np.unique(schema_type):
            # Deconvolve event representations from 6 stories
            train = (schema_type == schema) * (story_run != loro)
            all_design = np.concatenate(design_intact[train], axis=0)
            all_D = np.empty((dim, 0))
            for i in np.where(train)[0]:
                all_D = np.append(all_D, D[stories[i]].mean(2), axis=1)
            beta = zscore(deconv(all_D, all_design), axis=0, ddof=1)
            nEvents = beta.shape[1]

            # Create event model
            ev = brainiak.eventseg.event.EventSegment(nEvents)
            ev.set_event_patterns(beta)
            ev_var = ev.calc_weighted_event_var(all_D.T, all_design, beta)

            # Measure fit of event model to the 4 held-out stories
            test = np.where(story_run == loro)[0]
            for j in test:
                D_story = D[stories[j]]
                pred_labels = ev.find_events(D_story.mean(2).T, ev_var)[0]
                pred_mean = np.matmul(D_story.mean(2), pred_labels)
                cc = np.corrcoef(beta.T, pred_mean.T)[:nEvents, nEvents:]
                d = np.diag(cc).sum()
                # Mean of diagonal minus mean of off-diagonal
                evcorr[schema][j] = d/nEvents - \
                                    (cc.sum()-d)/(nEvents**2 - nEvents)

        # For pairs of held-out stories (one from each schema),
        # check whether they are correctly classified
        for lto_R in np.where((schema_type == 'R')*(story_run == loro))[0]:
            for lto_A in np.where((schema_type == 'A')*(story_run == loro))[0]:
                if (evcorr['R'][lto_R] + evcorr['A'][lto_A]) > \
                   (evcorr['R'][lto_A] + evcorr['A'][lto_R]):
                    acc[boot, 1] += 1/(len(np.unique(story_run))*2*2)

    print('    Accuracy = %.2f LTO, %.2f LORO' % (acc[boot, 0], acc[boot, 1]))

print('  ' + ROI + ' mean LTO accuracy = ' + str(acc[:, 0].mean()) +
      ', p = ' + str(norm.sf((acc[:, 0].mean()-0.5)/acc[:, 0].std())))
print('  ' + ROI + ' mean LORO accuracy = ' + str(acc[:, 1].mean()) +
      ', p = ' + str(norm.sf((acc[:, 1].mean()-0.5)/acc[:, 1].std())))

plt.figure()
plt.violinplot(acc[:, 0], showextrema=False, showmeans=True)
plt.plot([0.5, 1.5], [0.5, 0.5])
plt.ylim(0.2, 1)
plt.xticks([])
plt.ylabel('Classification accuracy')
plt.savefig('../results/Fig5_' + ROI + '.png')

plt.figure()
plt.violinplot(acc[:, 1], showextrema=False, showmeans=True)
plt.plot([0.5, 1.5], [0.5, 0.5])
plt.ylim(0.2, 1)
plt.xticks([])
plt.ylabel('Classification accuracy')
plt.savefig('../results/Fig5-1_' + ROI + '.png')
print(' ')
