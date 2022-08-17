import os
import socket
from datetime import datetime

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

host = socket.gethostname()

if 'node' in host:
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working'
else:
    raw_dir = '/Users/williams/Documents/phangs/jwst_data'
    working_dir = '/Users/williams/Documents/phangs/jwst_working'

date_str = datetime.today().strftime('%Y%m%d_%H:%M:%S')

log_file = os.path.join(working_dir,
                        'jwst_reprocess_log_%s.log' % date_str)

script_name = 'run_jwst_reprocessing.py'

os.system('python3 %s |& tee %s' % (script_name, log_file))

print('Complete!')
