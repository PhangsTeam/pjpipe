import copy
import glob

from astropy.io import fits
import os

os.chdir('/home/tgw/scratch/m33_reprocessed/v0p7/m33')

bands = [
    'F090W',
    'F090W_bgr',
    'F200W',
    'F200W_bgr',
    'F335M',
    'F335M_bgr',
    'F444W',
    'F444W_bgr',
    'F560W',
    'F560W_bgr',
    'F2100W',
    'F2100W_bgr',
]

for band in bands:
    align_files = glob.glob(os.path.join(band,
                                         'lv3',
                                         '*_i2d.fits'),
                            )

    with fits.open(align_files[0]) as hdu:
        sci_hdu = hdu['SCI']
        sci_hdu.writeto(f"{band}_testing.fits",
                        overwrite=True)

print('Complete!')
