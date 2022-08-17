import os
import socket
import glob

host = socket.gethostname()

if 'node' in host:
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working'
else:
    raw_dir = '/Users/williams/Documents/phangs/jwst_data'
    working_dir = '/Users/williams/Documents/phangs/jwst_working'

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
release_dir = os.path.join(working_dir, 'jwst_release')

version = 'v0p4'

reprocess_dir += '_%s' % version
release_dir = os.path.join(release_dir, version)

if not os.path.exists(release_dir):
    os.makedirs(release_dir)

galaxies = [
    'ngc0628',
    'ngc1365',
    # 'ngc7320',
    'ngc7496',
    'ic5332',
]

for galaxy in galaxies:

    print(galaxy)

    release_gal_dir = os.path.join(release_dir, galaxy)
    if not os.path.exists(release_gal_dir):
        os.makedirs(release_gal_dir)

    i2d_files = glob.glob(os.path.join(reprocess_dir,
                                       galaxy,
                                       '*',
                                       'lv3',
                                       '*_i2d.fits'))
    for i2d_file in i2d_files:
        os.system('cp %s %s' % (i2d_file, release_gal_dir))

    align_files = glob.glob(os.path.join(reprocess_dir,
                                         galaxy,
                                         '*',
                                         'lv3',
                                         '*_i2d_align.fits'))
    for align_file in align_files:
        os.system('cp %s %s' % (align_file, release_gal_dir))

print('Complete!')
