import os
import socket
import getpass

from nircam_destriping import NircamDestriper

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

galaxy = 'ngc0628'

hdu_name = os.path.join(galaxy,
                        'mastDownload',
                        'JWST',
                        'jw02107040001_02101_00001_nrcb3',
                        'jw02107040001_02101_00001_nrcb3_cal.fits'
                        )

nc_destripe = NircamDestriper(hdu_name=hdu_name,
                              destriping_method='row_median',
                              quadrants=False,
                              )
nc_destripe.run_destriping()

print('Complete!')
