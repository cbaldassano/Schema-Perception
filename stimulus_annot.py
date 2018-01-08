from deconvolve import regressor_to_TR
import numpy as np

stories = ['Brazil',  'Derek', 'MrBean',   'PulpFiction',
           'BigBang', 'Santa', 'Shame',    'Vinny',
           'DueDate', 'GLC',   'KandD',    'Nonstop',
           'Friends', 'HIMYM', 'Seinfeld', 'UpInTheAir']
nStories = len(stories)
schema_type = np.array(list('RRRRRRRRAAAAAAAA'))
modality = 'VVVVAAAAVVVVAAAA'
TR = 1.5

# Stimulus event boundaries (in seconds)
events_seconds = np.array([
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

# Stimulus lengths (in TRs)
story_TRs = np.array([124, 126, 125, 124,
                      115, 118, 118, 120,
                      127, 115, 124, 128,
                      110, 114, 116, 126])

# Compute design matrices for all stories
design = []
for i in range(len(stories)):
    subevent_E = np.zeros((events_seconds[i, 5], 5))
    for e in range(5):
        subevent_E[events_seconds[i, e]:(events_seconds[i, e + 1] + 1), e] = 1
    design.append(regressor_to_TR(subevent_E, TR, story_TRs[i]))
design = np.array(design)

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
