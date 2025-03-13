import pickle

import pandas as pd
import numpy as np
import nibabel as nb
import nitools as nt
import argparse

def make_demo_data_from_smp(sn=103, H='L', roi='S1'):

    # load ROI mask (in this case we want to look at S1)
    mask = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/ROI/subj{sn}/ROI.{H}.{roi}.nii')
    coords = nt.get_mask_coords(mask)

    # load regressor info
    reginfo = pd.read_csv(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/glm12/subj{sn}/subj{sn}_reginfo.tsv', sep='\t')

    # reorder reginfo for a more convenient visualization later
    reginfo['cue'] = reginfo['name'].str.extract(r'(?P<cue>\d+)%').astype(int)
    reginfo['stimFinger'] = reginfo['name'].str.extract(r',(?P<stimFinger>\w+)').fillna(0)
    reginfo = reginfo.sort_values(by=['run', 'stimFinger', 'cue', ], ascending=[True, True, True])
    cond_map = {
        '0%        ': 0,
        '25%       ': 1,
        '50%       ': 2,
        '75%       ': 3,
        '100%      ': 4,
        '25%,index ': 5,
        '50%,index ': 6,
        '75%,index ': 7,
        '100%,index': 8,
        '0%,ring   ': 9,
        '25%,ring  ': 10,
        '50%,ring  ': 11,
        '75%,ring  ': 12,

    }

    # define a condition vector and partition vector using reginfo
    cond_vec = reginfo['name'].map(cond_map)
    part_vec = reginfo['run']

    cond_names = [c.replace(' ', '') for c in cond_map.keys()]

    # load beta coefficients
    beta = np.zeros((reginfo.shape[0], coords.shape[1]))
    for nregr in range(reginfo.shape[0]):

        # load nifti file
        img = nb.load(f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/glm12/subj{sn}/beta_{nregr+1:04d}.nii')

        # apply roi mask
        beta[nregr] = nt.sample_image(img, coords[0], coords[1], coords[2], interpolation=0)

    # remove empty voxels
    beta = beta[reginfo.index][:, ~np.all(np.isnan(beta), axis=0)]

    beta = beta[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
    part_vec = part_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]
    cond_vec = cond_vec[(cond_vec == 0) | (cond_vec == 1) | (cond_vec == 2) | (cond_vec == 3) | (cond_vec == 4)]

    return beta, cond_vec, part_vec, cond_names[:5]


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

if __name__ == '__main__':
    main()