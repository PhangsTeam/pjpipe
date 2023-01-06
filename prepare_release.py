import os
import socket
import glob

host = socket.gethostname()

if 'node' in host:
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_phangs_reprocessed'
else:
    working_dir = '/Users/williams/Documents/phangs/jwst_reprocessed'

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
release_dir = os.path.join(working_dir, 'jwst_release')

version = 'v0p5'

reprocess_dir += '_%s' % version
release_dir = os.path.join(release_dir, version)

if not os.path.exists(release_dir):
    os.makedirs(release_dir)

galaxies = [
    'ngc0628',
    'ngc1365',
    'ngc1385'
    'ngc1566',
    # 'ngc7320',
    'ngc7496',
    'ic5332',
]

file_exts = ['i2d.fits',
             'i2d_align.fits',
             'i2d_align_table.fits',
             'cat.ecsv',
             'segm.fits']

for galaxy in galaxies:

    print(galaxy)

    release_gal_dir = os.path.join(release_dir, galaxy)
    if not os.path.exists(release_gal_dir):
        os.makedirs(release_gal_dir)

    for file_ext in file_exts:

        file_names = glob.glob(os.path.join(reprocess_dir,
                                            galaxy,
                                            '*',
                                            'lv3',
                                            '*_%s' % file_ext))
        for file_name in file_names:
            os.system('cp %s %s' % (file_name, release_gal_dir))

print('Complete!')
