import glob
import os
import sys

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
             'segm.fits']

working_dir = local['local']['working_dir']
version = config['pipeline']['data_version']
prop_ids = config['projects']

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
release_dir = os.path.join(working_dir, 'jwst_release')
reprocess_dir += '_%s' % version
release_dir = os.path.join(release_dir, version)

if not os.path.exists(release_dir):
    os.makedirs(release_dir)

for prop_id in prop_ids:

    targets = config['projects'][prop_id]['targets']

    for target in targets:

        print(target)

        release_target_dir = os.path.join(release_dir, target)
        if not os.path.exists(release_target_dir):
            os.makedirs(release_target_dir)

        for file_ext in file_exts:

            file_names = glob.glob(os.path.join(reprocess_dir,
                                                target,
                                                '*',
                                                'lv3',
                                                '*_%s' % file_ext))
            for file_name in file_names:
                os.system('cp %s %s' % (file_name, release_target_dir))

print('Complete!')
