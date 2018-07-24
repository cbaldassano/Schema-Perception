# Readme
This code repository demonstrates the analyses described in "Representation of real-world event schemas during narrative perception" by Baldassano, Hasson, and Norman. An executable version of the code (with ROI data) is available as [a Code Ocean capsule](https://codeocean.com/algorithm/a27d1d90-d227-4600-b876-051a801c7c20/).

Data files for the 8 ROIs are included in this capsule, for both the main (intact stimulus) experiment and the control (scrambled stimulus) experiment. Each is a set of (vertex x time x subject) matrices, one for each of the 16 stories used in the experiment. This data has been preprocessed using a surface-based pipeline.

The default run.sh runs all the ROI analyses from the paper.

## Fig 2 and 2-1 analysis
Fig2.py implements the event pattern correlation analysis for the main experiment (generating Figures 2 and 2-1 for one ROI). It has one command line argument: the name of the ROI to be analyzed.

## Fig 4 analysis
Fig4.py implements the event pattern correlation analysis for the scrambled experiment (generating Figure 5 for one ROI). It has one command line argument: the name of the ROI to be analyzed.

## Fig 5 and 5-1 analysis
Fig5.py implements the schema classification analysis (generating Figures 5 and 5-1 for one ROI). It has two command line arguments: the name of the ROI to be analyzed and the number of bootstrap samples to compute.

## Fig 6 analysis
Fig6.py implements the unsupervised alignment analysis (generating Figure 6 for one ROI). It has one command line argument: the name of the ROI to be analyzed.
