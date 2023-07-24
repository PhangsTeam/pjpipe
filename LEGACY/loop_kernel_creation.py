import itertools
import os
import socket

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

from make_jwst_kernels import MakeConvolutionKernel, get_pixscale

host = socket.gethostname()

if 'node' in host:
    webbpsf_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/webbpsf-data'
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_phangs_reprocessed/kernels'
else:
    webbpsf_path = '/Users/williams/Documents/webbpsf-data'
    base_dir = '/Users/williams/Documents/phangs/jwst_reprocessed/kernels'

os.environ['WEBBPSF_PATH'] = webbpsf_path

import webbpsf

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

psf_folder = os.path.join('../..', 'psfs')
if not os.path.exists(psf_folder):
    os.makedirs(psf_folder)

oversample_factor = 4
detector_oversample = 4
fov_arcsec = 20

overwrite = False

nircam_psfs = [
    # 'F200W',
    # 'F300M',
    # 'F335M',
    # 'F360M',
]

miri_psfs = [
    # 'F770W',
    # 'F1000W',
    # 'F1130W',
    'F2100W',
]

gauss_psfs = [
    # 'gauss0.73',
    # 'gauss1.66',
    'gauss1.0',
    'gauss1.3',
    'gauss1.67',
]
all_psfs = nircam_psfs + miri_psfs + gauss_psfs

nircam = webbpsf.NIRCam()
miri = webbpsf.MIRI()

# psf_pairs = [['F770W', 'F2100W'],
#              ['F770W', 'F1130W']]

psf_pairs = list(itertools.combinations(all_psfs, 2))

for psf_pair in psf_pairs:
    input_psf = psf_pair[0]
    output_psf = psf_pair[1]

    psf_type = {}

    if input_psf in nircam_psfs or input_psf in miri_psfs:
        input_wavelength = int(input_psf[1:-1])
        psf_type[input_psf] = 'jwst'
    elif input_psf in gauss_psfs:
        input_wavelength = 0
        psf_type[input_psf] = 'gauss'
    else:
        raise Warning('Unknown PSF type %s' % input_psf)

    if output_psf in nircam_psfs or output_psf in miri_psfs:
        output_wavelength = int(output_psf[1:-1])
        psf_type[output_psf] = 'jwst'
    elif output_psf in gauss_psfs:
        output_wavelength = 1e10
        psf_type[output_psf] = 'gauss'
    else:
        raise Warning('Unknown PSF type %s' % output_psf)

    if output_wavelength <= input_wavelength:
        continue

    output_file = os.path.join('%s_to_%s.fits' % (input_psf, output_psf))

    if os.path.exists(output_file) and not overwrite:
        continue

    print('Generating kernel from %s -> %s' % (input_psf, output_psf))

    psfs = {}

    for psf in [input_psf, output_psf]:

        psf_name = os.path.join(psf_folder, '%s.fits' % psf)

        if not os.path.exists(psf_name) or overwrite:
            # Create JWSTS PSFs. Use a detector_oversample of 1 so things are odd-shaped
            if psf in miri_psfs:
                miri.filter = psf
                psf_array = miri.calc_psf(oversample=oversample_factor,
                                          detector_oversample=detector_oversample,
                                          fov_arcsec=fov_arcsec,
                                          )[0]

            elif psf in nircam_psfs:
                nircam.filter = psf
                psf_array = nircam.calc_psf(oversample=oversample_factor,
                                            detector_oversample=detector_oversample,
                                            fov_arcsec=fov_arcsec,
                                            )[0]

            # Or build Gaussian PSF
            elif psf_type[psf] == 'gauss':
                fwhm = float(psf.strip('gauss'))
                pix_size = fwhm / 10
                std_pix = fwhm / 2.355 / pix_size

                gauss_2d = Gaussian2DKernel(x_stddev=std_pix)

                psf_array = fits.PrimaryHDU(data=np.array(gauss_2d.array, dtype=np.float32))

                psf_array.header['BITPIX'] = -32

                psf_array.header['CRPIX1'] = (gauss_2d.array.shape[1] + 1) / 2
                psf_array.header['CRPIX2'] = (gauss_2d.array.shape[0] + 1) / 2

                psf_array.header['CRVAL1'] = 0.00
                psf_array.header['CRVAL2'] = 0.00

                psf_array.header['CDELT1'] = - pix_size / 3600
                psf_array.header['CDELT2'] = pix_size / 3600

                psf_array.header['CTYPE1'] = 'RA---TAN'
                psf_array.header['CTYPE2'] = 'DEC--TAN'

            else:
                raise Warning('Not sure how to deal with PSF %s' % psf)

            # Pad if we're awkwardly even
            if psf_array.data.shape[0] % 2 == 0:
                new_data = np.zeros([psf_array.data.shape[0] + 1, psf_array.data.shape[1] + 1])
                new_data[:-1, :-1] = psf_array.data
                psf_array.data = new_data

            psf_array.writeto(psf_name, overwrite=True)
        else:
            psf_array = fits.open(psf_name)[0]

        psfs[psf] = psf_array

    common_pixscale = get_pixscale(psfs[output_psf])

    grid_size_arcsec = np.array([3645 * common_pixscale,
                                 3645 * common_pixscale])

    conv_kern = MakeConvolutionKernel(source_psf=psfs[input_psf],
                                      source_name=input_psf,
                                      target_psf=psfs[output_psf],
                                      target_name=output_psf,
                                      grid_size_arcsec=grid_size_arcsec,
                                      common_pixscale=common_pixscale,
                                      verbose=False,
                                      )
    conv_kern.make_convolution_kernel()
    conv_kern.write_out_kernel()

print('Complete!')
