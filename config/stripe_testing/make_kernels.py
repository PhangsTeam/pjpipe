import os

os.environ["WEBBPSF_PATH"] = '/data/beegfs/astro-storage/groups/schinnerer/williams/webbpsf-data'

from jwst_kernels.make_kernels import make_jwst_cross_kernel, make_jwst_kernel_to_Gauss

psf_dir = "/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data/psfs/"
out_dir = "/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data/kernels/"

if not os.path.exists(psf_dir):
    os.makedirs(psf_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

input_bands = [
    {"camera": "NIRCam", "filter": 'F150W'},
    {"camera": "NIRCam", "filter": 'F187N'},
    {"camera": "NIRCam", "filter": 'F212N'},
    {"camera": "NIRCam", "filter": 'F200W'},
    {"camera": "NIRCam", "filter": 'F277W'},
    {"camera": "NIRCam", "filter": 'F300M'},
    {"camera": "NIRCam", "filter": 'F335M'},
    {"camera": "NIRCam", "filter": 'F360M'},
    {"camera": "NIRCam", "filter": 'F430M'},
    {"camera": "NIRCam", "filter": 'F444W'},
]

target_bands = [
    {'camera': 'MIRI', 'filter': 'F2100W'},
]

gauss_bands = [
    {"fwhm": 0.85},
    {"fwhm": 0.9},
    {"fwhm": 1.0},
]

for ib in input_bands:
    for gb in gauss_bands:
        kk = make_jwst_kernel_to_Gauss(ib,
                                       gb,
                                       psf_dir=psf_dir,
                                       outdir=out_dir,
                                       )
        if gb["fwhm"] == 1:
            in_file = os.path.join(out_dir, f"{ib['filter'].lower()}_to_gauss1p00.fits")
            out_file = os.path.join(out_dir, f"{ib['filter'].lower()}_to_gauss1.fits")
            os.system(f"mv {in_file} {out_file}")

# for ib in input_bands:
#     for tb in target_bands:
#         kk = make_jwst_cross_kernel(ib,
#                                     tb,
#                                     psf_dir=psf_dir,
#                                     outdir=out_dir,
#                                     )
