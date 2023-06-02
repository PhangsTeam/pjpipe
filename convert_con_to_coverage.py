import glob
import os
import sys

import numpy as np
from astropy.io import fits
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if len(sys.argv) == 1:
    config_file = script_dir + '/config/config.toml'
    local_file = script_dir + '/config/local.toml'

elif len(sys.argv) == 2:
    local_file = script_dir + '/config/local.toml'
    config_file = sys.argv[1]

elif len(sys.argv) == 3:
    local_file = sys.argv[2]
    config_file = sys.argv[1]

else:
    raise Warning('Cannot parse %d arguments!' % len(sys.argv))

with open(config_file, 'rb') as f:
    config = tomllib.load(f)

with open(local_file, 'rb') as f:
    local = tomllib.load(f)

working_dir = local['local']['working_dir']
version = config['pipeline']['data_version']
prop_ids = config['projects']

release_dir = os.path.join(working_dir, 'jwst_release')
release_dir = os.path.join(release_dir, version)

os.chdir(release_dir)

for prop_id in tqdm(prop_ids, ascii=True, desc='Proposal IDs'):
    targets = config['projects'][prop_id]['targets']

    for target in tqdm(targets, ascii=True, desc='Targets', leave=False):

        # Pull out astronmetrically corrected images
        align_targets = glob.glob(os.path.join(target, '*_align.fits'))
        align_targets.sort()

        for align_target in align_targets:

            out_file = align_target.replace('_align.fits', '_coverage.fits')
            if os.path.exists(out_file):
                continue

            with fits.open(align_target, memmap=False) as hdu:

                # The data here comes as an int32, but needs to be uint32
                con_data = np.uint32(np.int32(hdu['CON'].data))
                header = hdu['SCI'].header

                coverage_map = np.zeros_like(hdu['SCI'].data)

                for plane in range(con_data.shape[0]):
                    con_plane = con_data[plane]

                    con_vals = np.unique(con_plane)

                    # Convert decimal to binary, so we can figure out the number of
                    # planes overlapping
                    bin_vals = [bin(con_val).replace('0b', '') for con_val in con_vals]
                    ns = [bin_val.count('1') for bin_val in bin_vals]

                    for (con_val, bin_val, n) in zip(con_vals, bin_vals, ns):
                        idx = np.where(con_plane == con_val)
                        coverage_map[idx] += n

                coverage_map[coverage_map == 0] = np.nan

                coverage_hdu = fits.ImageHDU(data=coverage_map, header=header)
                coverage_hdu.writeto(out_file, overwrite=True)
                del coverage_hdu
