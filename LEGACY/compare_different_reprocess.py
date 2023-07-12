import copy
import glob
import os
import sys
import warnings

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypdf import PdfMerger
from reproject import reproject_interp
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

script_dir = os.path.dirname(os.path.realpath(__file__))


def get_diff_image(filename,
                   v_curr,
                   v_prev,
                   percentiles=None,
                   ):

    if percentiles is None:
        percentiles = [1, 99]

    with fits.open(filename) as hdu1:

        prev_filename = filename.replace(v_curr, v_prev)
        if not os.path.exists(prev_filename):
            return None, None

        with fits.open(prev_filename) as hdu2:
            data1 = copy.deepcopy(hdu1['SCI'].data)
            data1[data1 == 0] = np.nan

            # Reproject the previous HDU
            hdu2['SCI'].data[hdu2['SCI'].data == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data2 = reproject_interp(hdu2['SCI'],
                                         hdu1['SCI'].header,
                                         return_footprint=False,
                                         )

            diff = data1 - data2

            v = np.nanmax(np.abs(np.nanpercentile(diff, percentiles)))

    return diff, v


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

# INPUT CURRENT AND PREVIOUS VERSIONS TO COMPARE HERE
curr_version = 'v0p8p1'
prev_version = 'v0p8'

working_dir = local['local']['working_dir']
curr_release_dir = os.path.join(working_dir, 'jwst_release', curr_version)
prev_release_dir = curr_release_dir.replace(curr_version, prev_version)

plot_dir = os.path.join(working_dir, 'comparison_plots', '%s_to_%s' % (curr_version, prev_version))
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

targets = os.listdir(curr_release_dir)

# Remove any weird hidden directories
targets = [target for target in targets if target[0] != '.']
targets.sort()

for target in tqdm(targets, ascii=True):
    jwst_files = glob.glob(os.path.join(curr_release_dir,
                                        target,
                                        '*_align.fits'))
    jwst_files.sort()

    file_dict = {}

    for key in ['nircam', 'miri']:
        files = [jwst_file for jwst_file in jwst_files
                 if key in os.path.split(jwst_file)[-1]
                 ]
        files.sort()
        file_dict[key] = files

    # Loop over and plot per-instrument
    for key in file_dict.keys():

        fancy_name = {'miri': 'MIRI',
                      'nircam': 'NIRCam',
                      }[key]

        files = copy.deepcopy(file_dict[key])
        if len(files) > 0:

            plot_name = os.path.join(plot_dir, '%s_%s_comparison' % (target, key))

            if os.path.exists('%s.png' % plot_name):
                continue

            plt.subplots(nrows=1, ncols=len(files), figsize=(4 * len(files), 4))

            for i, file in enumerate(files):
                band = os.path.split(file)[-1].split('_')[3]

                diff, v = get_diff_image(file,
                                         v_curr=curr_version,
                                         v_prev=prev_version,
                                         )

                ax = plt.subplot(1, len(files), i + 1)

                if diff is None:

                    plt.text(0.5, 0.5,
                             'Not present in %s' % prev_version,
                             ha='center', va='center',
                             fontweight='bold',
                             bbox=dict(fc='white', ec='black', alpha=0.9),
                             transform=ax.transAxes,
                             )

                    plt.axis('off')

                else:

                    vmin, vmax = -v, v

                    im = ax.imshow(diff,
                                   vmin=vmin, vmax=vmax,
                                   cmap=cmocean.cm.balance,
                                   origin='lower',
                                   interpolation='nearest',
                                   )

                    plt.text(0.05, 0.95,
                             band.upper(),
                             ha='left', va='top',
                             fontweight='bold',
                             bbox=dict(fc='white', ec='black', alpha=0.9),
                             transform=ax.transAxes,
                             )

                    plt.axis('off')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0)
                    plt.colorbar(im, cax=cax, label='MJy/sr')

            plt.suptitle('%s, %s' % (target.upper(), fancy_name))

            plt.tight_layout()

            plt.savefig('%s.png' % plot_name, bbox_inches='tight')
            plt.savefig('%s.pdf' % plot_name, bbox_inches='tight')
            plt.close()

# Merge these all into a single pdf doc
merged_filename = os.path.join(plot_dir,
                               '%s_to_%s_comparisons_merged.pdf' % (curr_version, prev_version))

pdfs = glob.glob(os.path.join(plot_dir,
                              '*_comparison.pdf',
                              )
                 )
pdfs.sort()

with PdfMerger() as merger:
    for pdf in pdfs:
        merger.append(pdf)

    merger.write(merged_filename)

print('Complete!')
