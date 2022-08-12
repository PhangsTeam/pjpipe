import os
import socket

from jwst_nircam_reprocess import NircamReprocess

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

reprocess_dir = os.path.join(working_dir, 'nircam_lev3_reprocessed')
crds_dir = os.path.join(working_dir, 'crds')

if flush_crds:
    os.system('rm -rf %s' % crds_dir)
    os.makedirs(crds_dir)

reprocess_dir_ext = 'v0p4'

reprocess_dir += '_%s' % reprocess_dir_ext

galaxies = [
    'ngc0628',
    'ngc1365',
    'ngc7320',
    'ngc7496',
]

for galaxy in galaxies:
    # alignment_fits = {'ngc0628': 'hlsp_phangs-hst_hst_acs-wfc_ngc628mosaic_f814w_v1_exp-drc-sci.fits',
    #                   'ngc1365': 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f814w_v1_exp-drc-sci.fits',
    #                   'ngc7320': 'hlsp_sm4ero_hst_wfc3_11502-ngc7318_f814w_v1_sci_drz.fits',
    #                   'ngc7496': 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_exp-drc-sci.fits',
    #                   }[galaxy]
    # alignment_image = os.path.join(working_dir,
    #                                'alignment_images',
    #                                alignment_fits)

    alignment_table_name = {'ngc0628': 'Gaia_DR3_NGC0628.fits',
                            'ngc1365': 'Gaia_DR3_NGC1365.fits',
                            'ngc7320': 'Gaia_DR3_NGC7320.fits',
                            'ngc7496': 'Gaia_DR3_NGC7496.fits',
                            }[galaxy]
    # alignment_table_name = '%s_agb_cat.fits' % galaxy
    alignment_table = os.path.join(working_dir,
                                   'alignment_images',
                                   alignment_table_name)

    if galaxy == 'ngc7320':
        alignment_mapping = {'F150W': 'F090W',
                             'F200W': 'F090W',
                             'F356W': 'F277W',
                             'F444W': 'F277W'}
        bands = ['F090W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
    else:
        alignment_mapping = {'F335M': 'F300M',
                             'F360M': 'F300M'}
        bands = ['F200W', 'F300M', 'F335M', 'F360M']

    nc_reproc = NircamReprocess(galaxy=galaxy,
                                raw_dir=raw_dir,
                                reprocess_dir=reprocess_dir,
                                crds_dir=crds_dir,
                                astrometric_alignment_type='table',
                                astrometric_alignment_table=alignment_table,
                                alignment_mapping=alignment_mapping,
                                bands=bands,
                                )
    nc_reproc.run_all()

print('Complete!')
