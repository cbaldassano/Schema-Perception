for ROI in Ang Aud hipp mPFC PHC PMC SFG STS
do
    python -u Fig2.py $ROI
    python -u Fig3.py $ROI 100
    python -u Fig4.py $ROI
done
