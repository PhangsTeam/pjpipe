import os
import socket
from datetime import datetime

from jwst_nircam_reprocess import NircamReprocess

host = socket.gethostname()

if 'node' in host:
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working'
else:
    raw_dir = '/Users/williams/Documents/phangs/jwst_data'
    working_dir = '/Users/williams/Documents/phangs/jwst_working'

date_str = datetime.today().strftime('%Y%m%d_%H:%M:%S')

log_file = os.path.join(working_dir,
                        'nircam_reprocess_log_%s.log' % date_str)

script_name = 'run_nircam_reprocessing.py'

os.system('python3 %s |& tee %s' % (script_name, log_file))

print('Complete!')
