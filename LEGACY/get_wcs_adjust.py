import copy

import numpy as np
import glob
import os
import sys
import warnings

from jwst_reprocess import FWHM_PIX, recursive_setattr, parse_parameter_dict

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

os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
os.environ['CRDS_PATH'] = local['local']['crds_dir']

from jwst import datamodels
from jwst.tweakreg import TweakRegStep

rad_to_arcsec = 3600 * np.rad2deg(1)

# We'll start from calibrated files just before lv3. These are in different
# places for NIRCam/MIRI

input_dir_dict = {
    'nircam': 'destripe',
    # 'nircam': 'cal',
    'miri': 'lyot_adjust',
}
input_ext = 'cal'
output_ext = 'wcs_adjust'

working_dir = local['local']['working_dir']
reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
reprocess_dir_ext = config['pipeline']['data_version']
reprocess_dir += '_%s' % reprocess_dir_ext

lv3_parameter_dict = config['lv3_parameters']
group_tweakreg_dithers = config['pipeline']['group_tweakreg_dithers']

prop_ids = config['projects']

os.chdir(reprocess_dir)

visit_transforms = {}

bands = config['pipeline']['miri_bands'] + config['pipeline']['nircam_bands']

for prop_id in prop_ids:

    targets = config['projects'][prop_id]['targets']

    for target in targets:

        visit_transforms[target] = {}

        for band in bands:

            if band in config['pipeline']['nircam_bands']:
                input_dir = input_dir_dict['nircam']
                band_type = 'nircam'
            else:
                input_dir = input_dir_dict['miri']
                band_type = 'miri'

            band_dir = os.path.join(target, band, input_dir)
            if not os.path.exists(band_dir):
                continue

            os.chdir(band_dir)

            fwhm_pix = FWHM_PIX[band]

            output_files = glob.glob('*%s.fits' % output_ext)

            if len(output_files) == 0:

                input_files = glob.glob('*%s.fits' % input_ext)
                input_files.sort()
                input_models = [datamodels.open(input_file) for input_file in input_files]
                asn_file = datamodels.ModelContainer(input_models)

                # Group up the dithers
                if band_type in group_tweakreg_dithers:
                    for model in asn_file._models:
                        model.meta.observation.exposure_number = '1'

                # If we only have one group, this won't do anything so just skip
                if len(asn_file.models_grouped) == 1:
                    os.chdir(reprocess_dir)
                    continue

                tweakreg_config = TweakRegStep.get_config_from_reference(asn_file)
                tweakreg = TweakRegStep.from_config_section(tweakreg_config)
                tweakreg.output_dir = os.getcwd()
                tweakreg.save_results = True
                tweakreg.suffix = output_ext
                tweakreg.kernel_fwhm = fwhm_pix * 2

                try:
                    tweakreg_params = lv3_parameter_dict['tweakreg']
                except KeyError:
                    tweakreg_params = {}

                for tweakreg_key in tweakreg_params:
                    value = parse_parameter_dict(tweakreg_params,
                                                 tweakreg_key,
                                                 band,
                                                 target,
                                                 )

                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(tweakreg, tweakreg_key, value)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    asn_file = tweakreg.run(asn_file)

            output_files = glob.glob('*%s.fits' % output_ext)
            output_files.sort()
            for output_file in output_files:

                # Get matrix and (x, y) shifts from the output file, if they exist
                aligned_model = datamodels.open(output_file)
                try:
                    transform = aligned_model.meta.wcs.forward_transform['tp_affine']
                    matrix = transform.matrix.value
                    xy_shift = rad_to_arcsec * transform.translation.value

                    visit = os.path.split(output_file)[-1].split('_')[0]
                    if visit in visit_transforms[target].keys():

                        visit_transforms[target][visit]['shift'] = np.vstack((visit_transforms[target][visit]['shift'],
                                                                              xy_shift)
                                                                             )
                        visit_transforms[target][visit]['matrix'] = np.dstack(
                            (visit_transforms[target][visit]['matrix'],
                             matrix)
                            )
                    else:
                        visit_transforms[target][visit] = {}
                        visit_transforms[target][visit]['shift'] = copy.deepcopy(xy_shift)
                        visit_transforms[target][visit]['matrix'] = copy.deepcopy(matrix)

                except IndexError:
                    pass

            os.chdir(reprocess_dir)

with open('wcs_adjust.toml', 'w+') as f:
    f.write('[wcs_adjust]\n\n')

    for target in visit_transforms.keys():

        # Skip where we don't have anything
        if len(visit_transforms[target].keys()) == 0:
            continue

        f.write('[wcs_adjust.%s]\n' % target)

        for visit in visit_transforms[target].keys():

            # If we only have one shift value, take that, otherwise take the mean
            if len(visit_transforms[target][visit]['shift'].shape) == 1:
                shift = visit_transforms[target][visit]['shift']
            else:
                shift = np.nanmean(visit_transforms[target][visit]['shift'], axis=0)

            # If we only have one matrix value, take that, otherwise take the mean
            if len(visit_transforms[target][visit]['matrix'].shape) == 2:
                matrix = visit_transforms[target][visit]['matrix']
            else:
                matrix = np.nanmean(visit_transforms[target][visit]['matrix'], axis=-1)

            # Format down to a few decimal points, to make things more readable
            f.write('%s.shift = %s\n' % (visit, [float('%.3f' % item) for item in shift]))
            f.write('%s.matrix = [\n\t%s,\n\t%s\n]\n' % (visit,
                                                         [float('%.3f' % item) for item in matrix[0]],
                                                         [float('%.3f' % item) for item in matrix[1]])
                    )

        f.write('\n')

print('Complete!')
