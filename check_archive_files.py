import glob
import os
import sys

from astropy.io import fits
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if len(sys.argv) == 1:
    local_file = script_dir + '/config/local.toml'

elif len(sys.argv) == 2:
    local_file = sys.argv[1]

else:
    raise Warning('Cannot parse %d arguments!' % len(sys.argv))

with open(local_file, 'rb') as f:
    local = tomllib.load(f)

raw_dir = local['local']['raw_dir']

os.chdir(raw_dir)

files = glob.glob(os.path.join('*',
                               '*',
                               '*',
                               '*',
                               '*.fits',
                               )
                  )
files.sort()

bad_files = []

for file in tqdm(files):

    try:
        hdu = fits.open(file, memmap=False)
        hdu.close()
    except:
        bad_files.append(file)

print('Found %d bad files' % len(bad_files))
for file in bad_files:
    print('-> %s ' % file)

print('Complete!')
