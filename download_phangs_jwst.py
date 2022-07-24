import os
import socket
import getpass

from archive_download import ArchiveDownload

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

prop_id = '2107'

targets = [
    'ngc7496',
    'ic5332',
    'ngc0628',
]

login = True
overwrite = True

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
