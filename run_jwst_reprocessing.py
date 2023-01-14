import glob

import os
import socket

from jwst_reprocess import JWSTReprocess
import sys

host = socket.gethostname()
script_dir = os.path.dirname(os.path.realpath(__file__))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if len(sys.argv) == 1:
    config_file = script_dir + '/config/config.toml'
    local_file = script_dir + '/config/local.toml'

if len(sys.argv) == 2:
    local_file = script_dir + '/config/local.toml'
    config_file = sys.argv[1]
    
if len(sys.argv) == 3:
    local_file = sys.argv[2]
    config_file = sys.argv[1]

with open(config_file,'rb') as f:
    config = tomllib.load(f)

with open(local_file,'rb') as f:
    local = tomllib.load(f)


raw_dir = local['local']['raw_dir']
working_dir = local['local']['working_dir']
updated_flats_dir = local['local']['updated_flats_dir']
if updated_flats_dir == '':
    updated_flats_dir = None

# We may want to occasionally flush out the CRDS directory to avoid weirdness between mappings. Probably do this at
# the start of another version cycle
flush_crds = config['pipeline']['flush_crds']

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
    for galaxy in targets:

        alignment_table_name = config['alignment'][galaxy]
        alignment_table = os.path.join(script_dir,
                                       'alignment',
                                       alignment_table_name)
        alignment_mapping = config['alignment_mapping']

        bands = (config['pipeline']['nircam_bands'] + 
                 config['pipeline']['miri_bands'])
        cur_field = config['pipeline']['lev3_fields']
        if cur_field == []:
            cur_field = None  


        reproc = JWSTReprocess(galaxy=galaxy,
                               raw_dir=raw_dir,
                               reprocess_dir=reprocess_dir,
                               crds_dir=crds_dir,
                               astrometric_alignment_type=config['lv3_parameters']['astrometric_alignment_type'],
                               astrometric_alignment_table=alignment_table,
                               alignment_mapping=alignment_mapping,
                               bands=bands,
                               do_all=True,
                               do_lv1=config['pipeline']['lv1'],
                               do_lv2=config['pipeline']['lv2'],
                               do_lv3=config['pipeline']['lv3'],
                               procs=local['local']['processors'],
                               overwrite_all=config['overwrite']['all'],
                               overwrite_lv1=config['overwrite']['lv1'],
                               overwrite_lv2=config['overwrite']['lv2'],
                               overwrite_lyot_adjust=config['overwrite']['lyot_adjust'],
                               overwrite_lv3=config['overwrite']['lv3'],
                               overwrite_astrometric_alignment=config['overwrite']['astrometric_alignment'],
                               overwrite_astrometric_ref_cat=config['overwrite']['astrometric_ref_cat'],
                               lv1_parameter_dict=config['lv1_parameters'],
                               lv2_parameter_dict=config['lv2_parameters'],
                               lv3_parameter_dict=config['lv3_parameters'],
                               updated_flats_dir=updated_flats_dir,
                               do_lyot_adjust=config['pipeline']['lyot_adjust'],
                               tpmatch_searchrad=config['tpmatch']['searchrad'],
                               tpmatch_separation=config['tpmatch']['separation'],
                               tpmatch_tolerance=config['tpmatch']['tolerance'],
                               tpmatch_use2dhist=config['tpmatch']['use2dhist'],
                               tpmatch_fitgeom=config['tpmatch']['fitgeom'],
                               tpmatch_nclip=config['tpmatch']['nclip'],
                               tpmatch_sigma=config['tpmatch']['sigma'],
                               # process_bgr_like_science=False,
                               use_field_in_lev3=cur_field
                               )
        reproc.run_all()



print('Complete!')
