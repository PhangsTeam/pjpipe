import os

import socket

from jwst_nircam_reprocess import NircamReprocess

host = socket.gethostname()

if 'node' in host:
    lv2_root_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/nircam_lev3_reprocessed'
    crds_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/crds'
else:
    lv2_root_dir = '/Users/williams/Documents/phangs/jwst_data'
    working_dir = '/Users/williams/Documents/phangs/jwst_working/nircam_lev3_reprocessed'
    crds_path = '/Users/williams/Documents/phangs/jwst_working/crds'

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

galaxies = [
    'ngc7496',
    'ngc0628',
]

for galaxy in galaxies:

    print('Reprocessing %s' % galaxy)

    nc_reproc = NircamReprocess(crds_path=crds_path,
                                galaxy=galaxy,
                                lv2_root_dir=lv2_root_dir,
                                working_dir=working_dir,
                                )
    nc_reproc.run_all()

print('Complete!')
