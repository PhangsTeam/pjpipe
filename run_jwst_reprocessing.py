import os
import socket

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from jwst_reprocess import JWSTReprocess

host = socket.gethostname()

if 'node' in host:
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working'
else:
    raw_dir = '/Users/williams/Documents/phangs/jwst_data'
    working_dir = '/Users/williams/Documents/phangs/jwst_working'

# We may want to occasionally flush out the CRDS directory to avoid weirdness between mappings. Probably do this at
# the start of another version cycle
flush_crds = False

reprocess_dir = os.path.join(working_dir, 'jwst_lv3_reprocessed')
crds_dir = os.path.join(working_dir, 'crds')

if flush_crds:
    os.system('rm -rf %s' % crds_dir)
    os.makedirs(crds_dir)

reprocess_dir_ext = 'v0p4'

reprocess_dir += '_%s' % reprocess_dir_ext

galaxies = [
    'ngc0628',
    'ngc1365',
    # 'ngc7320',
    'ngc7496',
    'ic5332',
]

for galaxy in galaxies:

    alignment_table_name = {
        'ngc0628': 'ngc0628_agb_cat.fits',
        'ngc1365': 'ngc1365_agb_cat.fits',
        'ngc7320': 'Gaia_DR3_NGC7320.fits',
        'ngc7496': 'ngc7496_agb_cat.fits'
    }[galaxy]
    alignment_table = os.path.join(working_dir,
                                   'alignment_images',
                                   alignment_table_name)

    if galaxy == 'ngc7320':
        alignment_mapping = {
            'F770W': 'F356W',
            'F1000W': 'F770W',
            'F1500W': 'F1000W'
        }

        bands = [
            # NIRCAM
            'F090W',
            'F150W',
            'F200W',
            'F277W',
            'F356W',
            'F444W',
            # MIRI
            'F770W',
            'F1000W',
            'F1500W',
        ]
    else:

        # We can't use NIRCAM bands for IC5332
        if galaxy in ['ic5532']:
            alignment_mapping = {
                'F1000W': 'F770W',  # Step up MIRI wavelengths
                'F1130W': 'F1000W',
                'F2100W': 'F1130W',
            }
        else:
            alignment_mapping = {
                'F770W': 'F335M',  # PAH->PAH
                'F1000W': 'F770W',  # Step up MIRI wavelengths
                'F1130W': 'F1000W',
                'F2100W': 'F1130W',
            }

        bands = [
            # NIRCAM
            'F200W',
            'F300M',
            'F335M',
            'F360M',
            # MIRI
            'F770W',
            'F1000W',
            'F1130W',
            'F2100W'
        ]

    reproc = JWSTReprocess(galaxy=galaxy,
                           raw_dir=raw_dir,
                           reprocess_dir=reprocess_dir,
                           crds_dir=crds_dir,
                           astrometric_alignment_type='table',
                           astrometric_alignment_table=alignment_table,
                           alignment_mapping=alignment_mapping,
                           bands=bands,
                           procs=20,
                           )
    reproc.run_all()

print('Complete!')
