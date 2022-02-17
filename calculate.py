import numpy
import cv2
import os.path
import csv

import ssim
import psnr

lanczos_files = os.listdir("inactive_faceenhancement_fivek_result/h6/hRHL2Q70_nofsr")
lanczos_files.sort()
GPEN_files = os.listdir("inactive_faceenhancement_fivek_result/h6/hGPENL2Q70_nofsr")
GPEN_files.sort()
quality_values = []
ssim_values = []
psnr_values = []

for i in range(0, len(GPEN_files)):    
    l = cv2.imread('inactive_faceenhancement_fivek_result/h6/hRHL2Q70_nofsr/' + lanczos_files[i])
    g = cv2.imread('inactive_faceenhancement_fivek_result/h6/hGPENL2Q70_nofsr/' + GPEN_files[i])
    
    Lan = l.astype(numpy.float32)
    GPEN = g.astype(numpy.float32)

    quality_values.append(i)
    ssim_values.append( ssim.ssim_exact(Lan/255, GPEN/255) )
    psnr_values.append( psnr.psnr(Lan, GPEN) )

with open('hRHL2Q70_nofsr_hGPENL2Q70_nofsr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ssim', ssim_values])
        writer.writerow(['psnr', psnr_values])