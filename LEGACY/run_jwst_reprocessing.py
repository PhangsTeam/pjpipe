import os
import sys

from LEGACY.jwst_reprocess import JWSTReprocess

script_dir = os.path.dirname(os.path.realpath(__file__))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

bands = None
if len(sys.argv) == 1:
    config_file = script_dir + '/config/config.toml'
    local_file = script_dir + '/config/local.toml'

elif len(sys.argv) == 2:
    local_file = script_dir + '/config/local.toml'
    config_file = sys.argv[1]

elif len(sys.argv) == 3:
    local_file = sys.argv[2]
    config_file = sys.argv[1]

elif len(sys.argv) == 4:
    local_file = sys.argv[2]
    config_file = sys.argv[1]
    bands = [sys.argv[3]]

else:
    raise Warning('Cannot parse %d arguments!' % len(sys.argv))

with open(config_file, 'rb') as f:
    config = tomllib.load(f)

with open(local_file, 'rb') as f:
    local = tomllib.load(f)

raw_dir = local['local']['raw_dir']
working_dir = local['local']['working_dir']
updated_flats_dir = local['local']['updated_flats_dir']
if updated_flats_dir == '':
    updated_flats_dir = None

if 'webb_psf_data' in local['local'].keys():
    webb_psf_dir = local['local']['webb_psf_data']
else:
    webb_psf_dir = None

# We may want to occasionally flush out the CRDS directory to avoid weirdness between mappings. Probably do this at
# the start of another version cycle
flush_crds = config['pipeline']['flush_crds']

processors = local['local']['processors']
if processors == '':
    processors = None

if 'pmap' in config['pipeline']['crds_context']:
    os.environ['CRDS_CONTEXT'] = config['pipeline']['crds_context']

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')

crds_dir = local['local']['crds_dir']
if crds_dir == '':
    crds_dir = os.path.join(working_dir, 'crds')
if not os.path.exists(crds_dir):
    os.makedirs(crds_dir)

if flush_crds:
    os.system('rm -rf %s' % crds_dir)
    os.makedirs(crds_dir)

reprocess_dir_ext = config['pipeline']['data_version']

reprocess_dir += '_%s' % reprocess_dir_ext

if 'api_key' in local['local'].keys():
    os.environ['MAST_API_TOKEN'] = local['local']['api_key']

# Load in WCS adjusts, if we have them
wcs_adjust_filename = os.path.join(reprocess_dir, 'wcs_adjust.toml')
if os.path.exists(wcs_adjust_filename):
    with open(wcs_adjust_filename, 'rb') as f:
        wcs_adjust = tomllib.load(f)
    wcs_adjust_dict = wcs_adjust['wcs_adjust']
else:
    wcs_adjust_dict = None

if 'alignment_dir' in local['local'].keys():
    alignment_dir = local['local']['alignment_dir']
else:
    alignment_dir = os.path.join(script_dir, 'alignment')


prop_ids = config['projects']
for prop_id in prop_ids:
    targets = config['projects'][prop_id]['targets']
    for target in targets:

        alignment_table_name = config['alignment'][target]
        alignment_table = os.path.join(alignment_dir,
                                       alignment_table_name)

        if bands is None:
            bands = (config['pipeline']['nircam_bands'] +
                     config['pipeline']['miri_bands'])

        cur_field = config['pipeline']['lev3_fields']
        if cur_field == []:
            cur_field = None

        reproc = JWSTReprocess(target=target,
                               raw_dir=raw_dir,
                               reprocess_dir=reprocess_dir,
                               crds_dir=crds_dir,
                               webb_psf_dir=webb_psf_dir,
                               bands=bands,
                               steps=config['pipeline']['steps'],
                               overwrites=config['pipeline']['overwrites'],
                               obs_to_skip=config['pipeline']['obs_to_skip'],
                               extra_obs_to_include=config['extra_obs_to_include'],
                               lv1_parameter_dict=config['lv1_parameters'],
                               lv2_parameter_dict=config['lv2_parameters'],
                               lv3_parameter_dict=config['lv3_parameters'],
                               bg_sub_parameter_dict=config['bg_sub_parameters'],
                               destripe_parameter_dict=config['destripe_parameters'],
                               dither_stripe_sub_parameter_dict=config['dither_stripe_sub_parameters'],
                               dither_match_parameter_dict=config['dither_match_parameters'],
                               astrometric_catalog_parameter_dict=config['astrometric_catalog_parameters'],
                               astrometry_parameter_dict=config['astrometry_parameters'],
                               psf_model_parameter_dict=config['psf_model_parameters'],
                               psf_model_dict=config['psf_model'],
                               wcs_adjust_dict=wcs_adjust_dict,
                               lyot_method=config['pipeline']['lyot_method'],
                               tweakreg_create_custom_catalogs=config['pipeline']['tweakreg_create_custom_catalogs'],
                               group_tweakreg_dithers=config['pipeline']['group_tweakreg_dithers'],
                               group_skymatch_dithers=config['pipeline']['group_skymatch_dithers'],
                               degroup_skymatch_dithers=config['pipeline']['degroup_skymatch_dithers'],
                               bgr_check_type=config['pipeline']['bgr_check_type'],
                               bgr_background_name=config['pipeline']['bgr_background_name'],
                               bgr_observation_types=config['pipeline']['bgr_observation_types'],
                               astrometric_alignment_type=config['pipeline']['astrometric_alignment_type'],
                               astrometric_alignment_table=alignment_table,
                               alignment_mapping=config['alignment_mapping'],
                               alignment_mapping_mode=config['pipeline']['alignment_mapping_mode'],
                               procs=processors,
                               updated_flats_dir=updated_flats_dir,
                               # process_bgr_like_science=False,
                               use_field_in_lev3=cur_field,
                               )
        reproc.run_all()

print('Complete!')
