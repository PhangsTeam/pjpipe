import os
import socket
from datetime import datetime
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

script_dir = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) == 1:
    config_file = script_dir + '/config/config.toml'
    local_file = script_dir + '/config/local.toml'

elif len(sys.argv) == 2:
    config_file = sys.argv[1]
    local_file = script_dir + '/config/local.toml'

elif len(sys.argv) == 3:
    config_file = sys.argv[1]
    local_file = sys.argv[2]

else:
    raise Warning('Cannot parse %d arguments!' % len(sys.argv))

with open(local_file, 'rb') as f:
    local = tomllib.load(f)

date_str = datetime.today().strftime('%Y%m%d_%H:%M:%S')

working_dir = local['local']['working_dir']

log_file = os.path.join(working_dir,
                        'jwst_reprocess_log_%s.log' % date_str)

script_name = 'run_jwst_reprocessing.py'

os.system('python3 %s %s %s |& tee %s' % (script_name, config_file, local_file, log_file))

print('Complete!')
