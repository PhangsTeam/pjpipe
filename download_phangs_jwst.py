import os
import socket
import getpass

from archive_download import ArchiveDownload

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_raw'
else:
    base_dir = '/Users/williams/Documents/jwst_raw'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

login = True
overwrite = False

product_type = [
    'SCIENCE',
    'PREVIEW',
    'INFO',
    'AUXILIARY',
]

calib_level = [1, 2, 3]

if login:
    api_key = getpass.getpass('Input API key: ')
else:
    api_key = None

prop_ids = [
    '2107',
]

for prop_id in prop_ids:

    targets = {
        '2107': [
            'ic5332',
            'ngc0628',
            'ngc1365',
            'ngc7496',
        ],
    }[prop_id]

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
