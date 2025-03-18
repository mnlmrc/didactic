import pickle

import pandas as pd
import numpy as np
import nibabel as nb
import nitools as nt
from nitools import spm
import argparse

# def make_demo_data_from_smp(sn=103, H='L', roi='S1'):
#
#     # load ROI mask (in this case we want to look at S1)
#     mask = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/ROI/subj{sn}/ROI.{H}.{roi}.nii')
#     coords = nt.get_mask_coords(mask)
#
#     cond_names = [c.replace(' ', '') for c in cond_map.keys()]
#
#     # load beta coefficients
#
#     # beta = np.zeros((reginfo.shape[0], coords.shape[1]))
#     # for nregr in range(reginfo.shape[0]):
#     #
#     #     # load nifti file
#     #     img = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/glm12/subj{sn}/beta_{nregr+1:04d}.nii')
#     #
#     #     # apply roi mask
#     #     beta[nregr] = nt.sample_image(img, coords[0], coords[1], coords[2], interpolation=0)
#
#     # load residual mean square
#     img = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/glm12/subj{sn}/ResMS.nii')
#     ResMS = nt.sample_image(img, coords[0], coords[1], coords[2], interpolation=0)
#
#     # remove empty voxels
#     beta = beta[reginfo.index][:, ~np.all(np.isnan(beta), axis=0)]
#     ResMS = ResMS[~np.isnan(ResMS)]
#
#     beta = beta[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
#     part_vec = part_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
#     cond_vec = cond_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
#
#     return beta, ResMS, cond_vec, part_vec, cond_names[:5]


def make_demo_data_from_smp(sn=103, H='L', roi='S1'):

    SPM = spm.SpmGlm(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/glm12/subj{sn}/')
    SPM.get_info_from_spm_mat()

    mask = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/ROI/subj{sn}/ROI.{H}.{roi}.nii')

    residuals, beta, info = SPM.get_residuals(mask)

    ResMS = np.square(residuals).mean(axis=0)

    cond_map = {
        '0%*bf(1)': 0,
        '25%*bf(1)': 1,
        '50%*bf(1)': 2,
        '75%*bf(1)': 3,
        '100%*bf(1)': 4,
        '25%,index*bf(1)': 5,
        '50%,index*bf(1)': 6,
        '75%,index*bf(1)': 7,
        '100%,index*bf(1)': 8,
        '0%,ring*bf(1)': 9,
        '25%,ring*bf(1)': 10,
        '50%,ring*bf(1)': 11,
        '75%,ring*bf(1)': 12,

    }

    cond_names = [c.split('*')[0] for c in cond_map.keys()]

    # define a condition vector and partition vector using reginfo
    cond_vec = np.vectorize(cond_map.get)(info['reg_name'])
    part_vec = info['run_number']

    beta = beta[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
    part_vec = part_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
    cond_vec = cond_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]

    return beta, ResMS, residuals, cond_vec, part_vec, cond_names[:5]



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', default='make_demo_data')
    parser.add_argument('--sn', type=int, default=103)
    parser.add_argument('--H', type=str, default='L')
    parser.add_argument('--roi', type=str, default='S1')

    args = parser.parse_args()

    if args.what == 'make_demo_data':
        Data = make_demo_data_from_smp(sn=args.sn, H=args.H, roi=args.roi)

        print('saving demo data...')
        f = open('data_demo_smp.p', 'wb')
        pickle.dump(Data, f)

    # if args.what == 'make_demo_residuals':
    #     residuals = make_demo_residuals(sn=args.sn, H=args.H, roi=args.roi)
    #
    #     print('saving residuals...')
    #     np.save('residuals.npy', residuals)

if __name__ == '__main__':
    main()