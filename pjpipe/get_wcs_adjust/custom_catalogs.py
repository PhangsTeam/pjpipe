import logging
import os

import numpy as np
from photutils.detection import IRAFStarFinder

from ..astrometric_catalog.constrained_diffusion import constrained_diffusion

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def constrained_diffusion_catalog(datamodel,
                                  output_directory,
                                  pivot_scale=3,
                                  snr_threshold=5,
                                  roundhi=0.1,
                                  roundlo=0.0,
                                  sharplo=0.8,
                                  sharphi=1.1,
                                  max_sources=1000,
                                  ):
    log.info(f"Generating custom catalog for {datamodel.meta.filename} with constrained diffusion.")
    img = datamodel.data
    noiseimg = datamodel.err
    noise = np.median(noiseimg[noiseimg > 0])

    decomposition, _ = constrained_diffusion(img, n_scales=4)
    smallscale = decomposition[0:pivot_scale, :, :].sum(axis=0)
    starfinder = IRAFStarFinder(snr_threshold * noise, 2.0, roundhi=roundhi, roundlo=roundlo,
                                sharplo=sharplo, sharphi=sharphi)
    sourcecat = starfinder(smallscale)

    if len(sourcecat) > max_sources:
        log.info(f"Found {len(sourcecat)} sources, limiting to {max_sources} brightest.")
        idx = np.argsort(sourcecat['flux'])[::-1]
        sourcecat = sourcecat[idx[:max_sources]]

    filename = datamodel.meta.filename.replace('.fits', '_condiff_sources.fits')
    fullfile = os.path.join(output_directory, filename)
    datamodel.meta.tweakreg_catalog = fullfile
    log.info(f"Writing custom catalog to {filename}")
    sourcecat.write(fullfile, overwrite=True)

    return datamodel
