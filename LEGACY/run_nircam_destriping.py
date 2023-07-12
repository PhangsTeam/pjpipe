import glob
import os
import socket
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

from tqdm import tqdm

set_start_method('fork')

from LEGACY.nircam_destriping import NircamDestriper


def parallel_destripe(hdu_name,
                      quadrants=True,
                      destriping_method='row_median',
                      sigma=3,
                      npixels=3,
                      max_iters=20,
                      dilate_size=11,
                      median_filter_scales=None,
                      pca_components=50,
                      pca_reconstruct_components=10,
                      pca_diffuse=False,
                      pca_dir=None,
                      just_sci_hdu=False,
                      out_dir=None,
                      plot_dir=None,
                      ):

    out_file = os.path.join(out_dir, os.path.split(hdu_name)[-1].replace('.fits', '_%s.fits' % destriping_method))

    if os.path.exists(out_file):
        return True

    if pca_dir is not None:
        pca_file = os.path.join(pca_dir,
                                os.path.split(hdu_name)[-1].replace('.fits', '_pca.pkl')
                                )
    else:
        pca_file = None

    nc_destripe = NircamDestriper(hdu_name=hdu_name,
                                  hdu_out_name=out_file,
                                  quadrants=quadrants,
                                  destriping_method=destriping_method,
                                  sigma=sigma,
                                  npixels=npixels,
                                  max_iters=max_iters,
                                  dilate_size=dilate_size,
                                  median_filter_scales=median_filter_scales,
                                  pca_components=pca_components,
                                  pca_reconstruct_components=pca_reconstruct_components,
                                  pca_diffuse=pca_diffuse,
                                  pca_file=pca_file,
                                  just_sci_hdu=just_sci_hdu,
                                  plot_dir=plot_dir,
                                  )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nc_destripe.run_destriping()

    return True


host = socket.gethostname()

if 'node' in host:
    # base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/nircam_lev3_reprocessed_v0p2/'
    working_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_working/destripe_tests'
else:
    # base_dir = '/Users/williams/Documents/phangs/jwst_data'
    base_dir = '/Users/williams/Documents/phangs/jwst_working/nircam_lev3_reprocessed_v0p2/'
    working_dir = '/Users/williams/Documents/phangs/jwst_working/destripe_tests'

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

procs = cpu_count() // 2

galaxies = [
    'ngc0628',
    'ngc7496',
]
bands = [
    'F200W',
    'F300M',
    'F335M',
    'F360M'
]

methods = [
    # 'pca',
    'pca+median',
    'median_filter'
]

plot_dir = os.path.join(working_dir,
                        'plots')

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pca_dir = os.path.join(working_dir,
                       'pca')

if not os.path.exists(pca_dir):
    os.makedirs(pca_dir)

if not os.path.exists(os.path.join(working_dir, 'bgsub')):
    os.makedirs(os.path.join(working_dir, 'bgsub'))

median_filter_scales = [1, 3, 7, 31, 63, 127, 511, 1023]

for galaxy in galaxies:

    print(galaxy)

    for band in bands:

        print(band)

        data_dir = os.path.join(base_dir,
                                galaxy,
                                band,
                                'cal')

        # hdu_names = os.path.join(base_dir,
        #                          galaxy,
        #                          'mastDownload',
        #                          'JWST',
        #                          'jw02107040001_02101_00001_nrcb3',
        #                          'jw02107040001_02101_00001_nrcb3_cal.fits'
        #                          )

        hdu_names = glob.glob(os.path.join(data_dir, '*_cal.fits'))
        hdu_names.sort()

        # hdu_names = glob.glob(os.path.join(data_dir, 'jw02107040001_02101_00001_nrcb4_cal.fits'))

        for method in methods:

            quadrants = {'pca': True,
                         'pca+median': True,
                         'median_filter': False,
                         }[method]

            print(method)

            with Pool(procs) as pool:

                results = []

                for result in tqdm(pool.imap_unordered(partial(parallel_destripe,
                                                               destriping_method=method,
                                                               median_filter_scales=median_filter_scales,
                                                               sigma=3,
                                                               dilate_size=7,
                                                               just_sci_hdu=True,
                                                               quadrants=quadrants,
                                                               pca_diffuse=True,
                                                               pca_reconstruct_components=5,
                                                               out_dir=os.path.join(working_dir, 'bgsub'),
                                                               pca_dir=pca_dir,
                                                               plot_dir=plot_dir,
                                                               ),
                                                       hdu_names),
                                   total=len(hdu_names), ascii=True):
                    results.append(result)

print('Complete!')
