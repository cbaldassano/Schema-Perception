for ROI in Ang Aud hipp mPFC PHC PMC SFG STS
do
    python -u Fig2.py $ROI
    python -u Fig4.py $ROI
    python -u Fig5.py $ROI 100
done

for ROI in mPFC PMC SFG
do
    python -u Fig6.py $ROI
done

