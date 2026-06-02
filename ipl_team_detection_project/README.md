# IPL Team Detection Using Traditional Machine Learning

## Overview

This project detects IPL cricket teams in image grid cells using:

- OpenCV
- HOG Features
- HSV Color Histograms
- LBP Texture Features
- Random Forest
- XGBoost
- SVM
- KNN

No Deep Learning or CNNs are used.

---

## Dataset Format

Images:
- Size: 800x600

Grid:
- 8x8
- 64 cells

Labels:

0 = No Team
1 = CSK
2 = DC
3 = GT
4 = KKR
5 = LSG
6 = MI
7 = PBKS
8 = RR
9 = RCB
10 = SRH

---

## Setup

```bash
pip install -r requirements.txt