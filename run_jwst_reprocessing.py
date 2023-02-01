import os
import sys

from jwst_reprocess import JWSTReprocess

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

raw_dir = local['local']['raw_dir']
working_dir = local['local']['working_dir']
updated_flats_dir = local['local']['updated_flats_dir']
if updated_flats_dir == '':
    updated_flats_dir = None

# We may want to occasionally flush out the CRDS directory to avoid weirdness between mappings. Probably do this at
# the start of another version cycle
flush_crds = config['pipeline']['flush_crds']

processors = local['local']['processors']
if processors == '':
    processors = None

if 'pmap' in config['pipeline']['crds_context']:
    os.environ['CRDS_CONTEXT'] = config['pipeline']['crds_context']

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
crds_dir = os.path.join(working_dir, 'crds')

if flush_crds:
    os.system('rm -rf %s' % crds_dir)
    os.makedirs(crds_dir)

reprocess_dir_ext = config['pipeline']['data_version']

reprocess_dir += '_%s' % reprocess_dir_ext

prop_ids = config['projects']
for prop_id in prop_ids:
    targets = config['projects'][prop_id]['targets']
    for target in targets:

        alignment_table_name = config['alignment'][target]
        alignment_table = os.path.join(script_dir,
                                       'alignment',
                                       alignment_table_name)
        alignment_mapping = config['alignment_mapping']

        bands = (config['pipeline']['nircam_bands'] +
                 config['pipeline']['miri_bands'])
        cur_field = config['pipeline']['lev3_fields']
        if cur_field == []:
            cur_field = None

        reproc = JWSTReprocess(target=target,
                               raw_dir=raw_dir,
                               reprocess_dir=reprocess_dir,
                               crds_dir=crds_dir,
                               bands=bands,
                               steps=config['pipeline']['steps'],
                               overwrites=config['pipeline']['overwrites'],
                               lv1_parameter_dict=config['lv1_parameters'],
                               lv2_parameter_dict=config['lv2_parameters'],
                               lv3_parameter_dict=config['lv3_parameters'],
                               astrometry_parameter_dict=config['astrometry_parameters'],
                               lyot_method=config['pipeline']['lyot_method'],
                               astrometric_alignment_type='table',
                               astrometric_alignment_table=alignment_table,
                               alignment_mapping=alignment_mapping,
                               procs=processors,
                               updated_flats_dir=updated_flats_dir,
                               # process_bgr_like_science=False,
                               use_field_in_lev3=cur_field,
                               )
        reproc.run_all()

print('Complete!')
