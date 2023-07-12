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

file_exts = ['i2d.fits',
             'i2d_align.fits',
             'i2d_align_table.fits',
             'cat.ecsv',
             'astro_cat.fits',
             'segm.fits',
             ]

tweakback_ext = 'tweakback.fits'

working_dir = local['local']['working_dir']
version = config['pipeline']['data_version']
prop_ids = config['projects']

remove_bloat = config['prepare_release']['remove_bloat']
move_tweakback = config['prepare_release']['move_tweakback']
overwrite = config['prepare_release']['overwrite']

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
release_dir = os.path.join(working_dir, 'jwst_release')
reprocess_dir += '_%s' % version
release_dir = os.path.join(release_dir, version)

if not os.path.exists(release_dir):
    os.makedirs(release_dir)

hdu_ext_to_delete = [
    # 'ERR',
    # 'CON',
    # 'WHT',
    'VAR_POISSON',
    'VAR_RNOISE',
    'VAR_FLAT',
]

for prop_id in tqdm(prop_ids, ascii=True, desc='prop ids'):

    targets = config['projects'][prop_id]['targets']

    for target in tqdm(targets, ascii=True, leave=False, desc='targets'):

        release_target_dir = os.path.join(release_dir, target)
        if not os.path.exists(release_target_dir):
            os.makedirs(release_target_dir)

        for file_ext in tqdm(file_exts, ascii=True, leave=False, desc='main files'):

            file_names = glob.glob(os.path.join(reprocess_dir,
                                                target,
                                                '*',
                                                'lv3',
                                                '*_%s' % file_ext))
            file_names.sort()

            for file_name in file_names:

                hdu_out_name = os.path.join(release_target_dir, os.path.split(file_name)[-1])

                if not os.path.exists(hdu_out_name) or overwrite:

                    if file_ext in ['i2d.fits', 'i2d_align.fits'] and remove_bloat:
                        # For these, we want to pull out only the data and error extensions. Everything else
                        # is just bloat

                        hdu = fits.open(file_name)
                        for hdu_ext in hdu_ext_to_delete:
                            del hdu[hdu_ext]

                        out_name = os.path.join(release_target_dir, os.path.split(file_name)[-1])
                        hdu.writeto(out_name, overwrite=True)
                        hdu.close()

                    else:

                        os.system('cp %s %s' % (file_name, release_target_dir))

        # Also move the tweakback files, but put these into a separate directory to avoid too much clutter

        if move_tweakback:
            tweakback_files = glob.glob(os.path.join(reprocess_dir,
                                                     target,
                                                     '*',
                                                     'lv3',
                                                     '*_%s' % tweakback_ext))
            tweakback_files.sort()

            for tweakback_file in tqdm(tweakback_files, ascii=True, leave=False, desc='tweakback'):

                band = tweakback_file.split(os.path.sep)[-3]
                tweakback_dir = os.path.join(release_target_dir, '%s_tweakback' % band.lower())
                if not os.path.exists(tweakback_dir):
                    os.makedirs(tweakback_dir)

                hdu_out_name = os.path.join(tweakback_dir, os.path.split(tweakback_file)[-1])

                if not os.path.exists(hdu_out_name) or overwrite:

                    os.system('cp %s %s' % (tweakback_file, tweakback_dir))

print('Complete!')
