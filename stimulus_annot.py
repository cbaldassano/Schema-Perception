from deconvolve import regressor_to_TR
import numpy as np

stories = ['Brazil',  'Derek', 'MrBean',   'PulpFiction',
           'BigBang', 'Santa', 'Shame',    'Vinny',
           'DueDate', 'GLC',   'KandD',    'Nonstop',
           'Friends', 'HIMYM', 'Seinfeld', 'UpInTheAir']
nStories = len(stories)
schema_type = np.array(list('RRRRRRRRAAAAAAAA'))
modality = np.array(list('VVVVAAAAVVVVAAAA'))
TR = 1.5

# Story event boundaries (in seconds)
events_seconds_intact = np.array([
     [0,     6,    46,    73,   115,   184],
     [0,     6,    30,   111,   169,   188],
     [0,     6,    21,    74,   159,   186],
     [0,     6,    40,    56,   171,   185],
     [0,     6,    31,    46,   156,   171],
     [0,     6,    71,   104,   165,   175],
     [0,     6,    26,    94,   134,   175],
     [0,     6,    28,    59,   151,   179],
     [0,     6,    51,   111,   122,   189],
     [0,     6,    23,    70,    84,   171],
     [0,     6,    53,    82,   136,   184],
     [0,     6,    29,    77,   123,   190],
     [0,     6,    30,    97,   138,   163],
     [0,     6,    36,   109,   124,   169],
     [0,     6,    39,    85,   107,   173],
     [0,     6,    33,   108,   144,   187]])

# Story lengths (in TRs)
story_TRs = np.array([124, 126, 125, 124,
                      115, 118, 118, 120,
                      127, 115, 124, 128,
                      110, 114, 116, 126])

# Labels of which stimuli were shown in the same run
story_run = np.array(list('ABCDABCDABCDABCD'))


# Condition mask for pairs of stories:
#   1 = Same story
#   2 = Same schema, same modality
#   3 = Same schema, diff modality
#   4 = Diff schema, same modality
#   5 = Diff schema, diff modality
mask = np.zeros((nStories, nStories))
for i in range(nStories):
    for j in range(nStories):
        if i == j:
            mask[i][j] = 1
        else:
            if schema_type[i] == schema_type[j]:
                if modality[i] == modality[j]:
                    mask[i][j] = 2
                else:
                    mask[i][j] = 3
            else:
                if modality[i] == modality[j]:
                    mask[i][j] = 4
                else:
                    mask[i][j] = 5

# Compute design matrices for all stories
design_intact = []
for i in range(len(stories)):
    subevent_E = np.zeros((events_seconds_intact[i, 5], 5))
    for e in range(5):
        subevent_E[events_seconds_intact[i, e]:
                   (events_seconds_intact[i, e + 1] + 1), e] = 1
    design_intact.append(regressor_to_TR(subevent_E, TR, story_TRs[i]))
design_intact = np.array(design_intact)


# Schemas and modalities of scrambled clips
clips = ['RV1', 'RV2', 'RV3', 'RV4',
         'RA1', 'RA2', 'RA3', 'RA4',
         'AV1', 'AV2', 'AV3', 'AV4',
         'AA1', 'AA2', 'AA3', 'AA4']

# Timing of events in scrambled clips
# Each row corresponds to the four events in one story
# Location of each event given as [clip number, start time, end time]
events_seconds_scrambled = np.array([
    [[1,  20,  62], [2,   6,  29], [4, 119, 162], [3, 113, 183]],
    [[2, 174, 198], [4,  21, 101], [3,   6,  64], [1,  62,  83]],
    [[4, 101, 119], [3,  64, 113], [1,  83, 167], [2,  29,  58]],
    [[3, 183, 215], [1,   6,  20], [2,  58, 174], [4,   6,  21]],
    [[1,  77, 101], [2,  77,  91], [3,  99, 211], [4,   6,  21]],
    [[3,  34,  99], [1,   6,  37], [4,  21,  83], [2,  67,  77]],
    [[4, 112, 133], [3, 211, 277], [2,   6,  46], [1,  37,  77]],
    [[2,  46,  67], [4,  83, 112], [1, 101, 194], [3,   6,  34]],
    [[2,  20,  64], [4, 102, 162], [1, 187, 199], [3,   6,  74]],
    [[4, 162, 177], [3, 151, 197], [2,   6,  20], [1,  53, 142]],
    [[1, 142, 187], [2, 132, 160], [3,  96, 151], [4,   6,  55]],
    [[3,  74,  96], [1,   6,  53], [4,  55, 102], [2,  64,  132]],
    [[1,  83, 111], [2,   6,  68], [4, 153, 194], [3,  66,  92]],
    [[2, 171, 199], [4,  48, 122], [3,   6,  20], [1, 111, 157]],
    [[4, 122, 153], [3,  20,  66], [1, 157, 175], [2,  68, 139]],
    [[3,  92, 119], [1,   6,  83], [2, 139, 171], [4,   6,  48]]])

# Clip lengths (in TRs)
clip_TRs = np.array([112, 133, 144, 109,
                     130,  62, 186,  90,
                     134, 108, 132, 119,
                     118, 134,  80, 130])

# Design matrices (four regressors each)
design_scrambled = []

# The story for each of the four regressors in design
design_scrambled_story = np.zeros((len(clips), 4), dtype=np.int)

# The event for each of the four regressors in design
design_scrambled_event = np.zeros((len(clips), 4), dtype=np.int)

for i in range(len(clips)):
    # Find the stories with events in this clip
    matching_stories = (schema_type == clips[i][0]) * \
                       (modality == clips[i][1])
    ev = np.transpose(np.nonzero(events_seconds_scrambled[:, :, 0] ==
                                 int(clips[i][2])))[:, 1]
    design_scrambled_event[i, :] = ev[matching_stories]
    design_scrambled_story[i, :] = np.arange(nStories)[matching_stories]

    # Construct regressors
    timing = events_seconds_scrambled[design_scrambled_story[i, :],
                                      design_scrambled_event[i, :]][:, 1:3]
    subevent_E = np.zeros((timing[:, 1].max(), 4))
    for e in range(4):
        subevent_E[timing[e, 0]:(timing[e, 1] + 1), e] = 1
    design_scrambled.append(regressor_to_TR(subevent_E, TR, clip_TRs[i]))
design_scrambled = np.array(design_scrambled)
