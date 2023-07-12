import os
import socket

from LEGACY.miri_destriping import MiriDestriper

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    out_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/destripe_tests'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst_data'
    out_dir = '//Users/williams/Documents/phangs/jwst_working/destripe_tests'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.chdir(base_dir)

galaxy = 'ngc0628'

hdu_name = os.path.join(galaxy,
                        'mastDownload',
                        'JWST',
                        'jw02107040002_08201_00004_mirimage',
                        'jw02107040002_08201_00004_mirimage_cal.fits'
                        )
hdu_out_name = os.path.join(out_dir, hdu_name.split(os.path.sep)[-1])

miri_destripe = MiriDestriper(hdu_name=hdu_name,
                              hdu_out_name=hdu_out_name,
                              destriping_method='median_filter',
                              median_filter_scales=[1, 7, 31, 63, 127, 511],
                              )
miri_destripe.run_destriping()

print('Complete!')
