import socket
from datetime import datetime

from jwst_nircam_reprocess import NircamReprocess

host = socket.gethostname()

if 'node' in host:
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    reprocess_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/nircam_lev3_reprocessed'
    crds_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/crds'
else:
    raw_dir = '/Users/williams/Documents/phangs/jwst_data'
    reprocess_dir = '/Users/williams/Documents/phangs/jwst_working/nircam_lev3_reprocessed_hdr_testing'
    crds_path = '/Users/williams/Documents/phangs/jwst_working/crds'

date_str = datetime.today().strftime('%Y%m%d')

reprocess_dir += '_%s' % date_str

galaxies = [
    'ngc0628',
    'ngc7496',
]

for galaxy in galaxies:
    print('Reprocessing %s' % galaxy)

    nc_reproc = NircamReprocess(crds_path=crds_path,
                                galaxy=galaxy,
                                raw_dir=raw_dir,
                                reprocess_dir=reprocess_dir,
                                do_all=True,
                                )
    nc_reproc.run_all()

print('Complete!')
