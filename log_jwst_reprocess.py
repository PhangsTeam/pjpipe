import os
import socket
from datetime import datetime

host = socket.gethostname()

if 'node' in host:
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_phangs_reprocessed'
else:
    working_dir = '/Users/williams/Documents/phangs/jwst_reprocessed'

date_str = datetime.today().strftime('%Y%m%d_%H:%M:%S')

log_file = os.path.join(working_dir,
                        'jwst_reprocess_log_%s.log' % date_str)

script_name = 'run_jwst_reprocessing.py'

os.system('python3 %s |& tee %s' % (script_name, log_file))

print('Complete!')
