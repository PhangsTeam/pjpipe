import os
import socket
import getpass

from archive_download import ArchiveDownload

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open('config.toml','rb') as f:
    config = tomllib.load(f)

with open('local.toml','rb') as f:
    local = tomllib.load(f)


host = socket.gethostname()

base_dir = local['local']['base_dir']
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



prop_ids = config['download']['projects']
targets = config['galaxies']['targets']

for prop_id in prop_ids:
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
