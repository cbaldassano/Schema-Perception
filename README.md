# Readme
This code repository demonstrates the analyses described in "Representation of real-world event schemas during narrative perception" by Baldassano, Hasson, and Norman.

To minimize execution time, the analyses are carried out for only one ROI and are not bootstrapped, as described below.

## Fig 2 analysis
Fig2.py implements the event pattern correlation analysis for one ROI. The full figure in the paper included 8 ROIs, and was also run on surface-based searchlights. The output is Fig2.png.

## Fig 3 analysis
Fig3.py implements the schema classification analysis. To generate the full bootstrap distribution in the paper, this analysis was repeated 100 times on different bootstrap samples of the original data (before applying SRM, which combines information across subjects). The output accuracy is printed to the terminal.

## Fig 4 analysis
Fig4.py implements the unsupervised alignment analysis. The full figure in the paper included 3 ROIs. The output is Fig4.png.
