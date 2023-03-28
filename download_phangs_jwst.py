import os
import sys

from archive_download import ArchiveDownload

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

script_dir = os.path.dirname(os.path.realpath(__file__))

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

base_dir = local['local']['raw_dir']
api_key = local['local']['api_key']

login = False
if api_key != '':
    login = True
else:
    try:
        api_key = os.environ['MAST_API_TOKEN']
        login = True
    except KeyError:
        pass

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

overwrite = False

product_type = config['download']['products']

calib_level = config['download']['calib_level']

prop_ids = config['projects']

for prop_id in prop_ids:
    targets = config['projects'][prop_id]['targets']
    for target in targets:
        dl_dir = target.replace(' ', '_')
        if not os.path.exists(dl_dir):
            os.makedirs(dl_dir)
        os.chdir(dl_dir)

        archive_dl = ArchiveDownload(target=target,
                                     prop_id=prop_id,
                                     login=login,
                                     api_key=api_key,
                                     calib_level=calib_level,
                                     product_type=product_type,
                                     overwrite=overwrite)
        archive_dl.archive_download()

        os.chdir(base_dir)

print('Complete!')
