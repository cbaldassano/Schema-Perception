# Readme
This code repository demonstrates the analyses described in "Representation of real-world event schemas during narrative perception" by Baldassano, Hasson, and Norman. An executable version of the code (with ROI data) is available as [a Code Ocean capsule](https://codeocean.com/2018/09/12/schema-perception/code).

The default run.sh runs all the ROI analyses from the paper.

## Fig 2 and 3 analysis
Fig2_3.py implements the event pattern correlation analysis for the main experiment (generating Figures 2 and 3 for one ROI). It has one command line argument: the name of the ROI to be analyzed.

## Fig 5 analysis
Fig5.py implements the event pattern correlation analysis for the scrambled experiment (generating Figure 5 for one ROI). It has one command line argument: the name of the ROI to be analyzed.

## Fig 6 analysis
Fig6.py implements the schema classification analysis (generating Figure 6 for one ROI). It has two command line arguments: the name of the ROI to be analyzed and the number of bootstrap samples to compute.

## Fig 7 analysis
Fig7.py implements the unsupervised alignment analysis (generating Figure 7 for one ROI). It has one command line argument: the name of the ROI to be analyzed.
