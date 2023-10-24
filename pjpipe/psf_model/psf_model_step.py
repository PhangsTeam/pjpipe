import copy
import gc
import glob
import logging
import os
import shutil
import warnings

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import webbpsf
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from lmfit import minimize, Parameters, fit_report
from photutils.centroids import centroid_com
from photutils.segmentation import detect_sources
from scipy.ndimage import shift
from stdatamodels.jwst import datamodels

from ..utils import get_dq_bit_mask, make_source_mask

matplotlib.use("agg")
log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

ALLOWED_METHODS = ["replace", "subtract"]


def get_sat_mask(dq):
    """Get mask of saturated values from DQ array

    This is a bit fiddly, but these are either flagged as
    3 (DO_NOT_USE+SATURATION) or 7 (DO_NOT_USE+SATURATION+DET_JUMP)

    Args:
        dq: Input DQ array
    """

    sat_mask = (dq == 3) | (dq == 7)

    return sat_mask


def image_resid(
        theta,
        data=None,
        err=None,
        psf=None,
        mask=None,
        x_cen=None,
        y_cen=None,
        psf_thresh=1e-5,
):
    """Scale and shift PSF, calculate residual.

    Args:
        theta: Parameters to fit
        data: Input data
        err: Input error
        psf: PSF to fit into the data
        mask: Optional mask to define good data
        x_cen: Initial guess for the x centre of the saturated source
        y_cen: Initial guess for the y centre of the saturated source
        psf_thresh: We only care about fitting in the region where
            the PSF is measurable. This defaults to 1e-5 (i.e. 0.001% of the
            maximum PSF amplitude)
    """

    if data is None:
        raise TypeError("data should be defined!")
    if psf is None:
        raise TypeError("psf should be defined")

    data = copy.deepcopy(data)
    err = copy.deepcopy(err)
    psf = copy.deepcopy(psf)

    amp = theta["amp"]
    x_shift = theta["x_cen"]
    y_shift = theta["y_cen"]
    offset = theta["offset"]

    if x_cen is not None:
        x_shift -= x_cen
    if y_cen is not None:
        y_shift -= y_cen

    # Now put this into a model, ensuring we've centroided to shift
    psf_x_cen, psf_y_cen = centroid_com(psf)

    model = np.zeros_like(data)

    model[: psf.shape[0], : psf.shape[1]] = psf

    # Now shift to the new coords
    model = shift(
        model,
        shift=[y_shift + y_cen - psf_y_cen, x_shift + x_cen - psf_x_cen],
    )

    # We mostly care in the region where the PSF is
    psf_mask = (model < psf_thresh * np.nanmax(psf)) & (np.isfinite(data))

    # Scale and offset the model
    model = amp * model + offset

    data[psf_mask] = 0
    model[psf_mask] = 0

    if mask is not None:
        data[mask] = np.nan
        if err is not None:
            err[mask] = np.nan
        model[mask] = np.nan

    resid = residual(
        data,
        model,
        err=err,
    )

    return resid


def residual(
        data,
        model,
        err=None,
):
    """Calculate residual for data (and optional error)

    Just runs a simple chi-calculation for LMFIT. If errors
    are included, will use those
    """

    # Filter NaNs
    good_idx = np.where(np.isfinite(data) & np.isfinite(model))
    data = copy.deepcopy(data[good_idx])
    model = copy.deepcopy(model[good_idx])

    if err is not None:
        err = copy.deepcopy(err[good_idx])

    resid = data - model

    if err is None:
        return resid
    else:
        return resid / err


class PSFModelStep:
    def __init__(
            self,
            in_dir,
            out_dir,
            step_ext,
            procs,
            method="replace",
            npixels=9,
            separation=0.1,
            psf_fov_pixels=511,
            psf_thresh=1e-5,
            dilate_size=7,
            nsigma=5,
            overwrite=False,
    ):
        """Step to model the PSF in saturated sources

        In the centres of galaxies, saturation and PSF wings can blow out the image
        in an unpleasant way. This step attempts to alleviate that by finding saturated
        sources and either subtracting the PSF, or painting in the saturated regions

        N.B. This is still highly preliminary, and should be seen as alpha. It hasn't
        been thoroughly tested across the whole sample yet, so weird errors may arise.
        You have been warned!

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel. Currently, does nothing
            method: Whether to "replace" saturated cores, or "subtract" the PSF.
                Defaults to replace
            npixels: Minimum number of pixels to define a saturated source. Defaults
                to 9
            separation: When creating catalogues for the saturated sources, this is the
                minimum distance (in arcsec) to identify a distinct source. Defaults to
                0.1
            psf_fov_pixels: Size of the simulated PSF. Should be odd so it has a centre.
                Defaults to 511
            psf_thresh: Minimum threshold to define where we consider the PSF to be significant
                (and thus used in the fit). Defaults to 1e-5, i.e. 0.001% of the PSF peak
            dilate_size: Dilate size for creating source mask before fitting, since we don't
                want to fit in very bright areas. Defaults to 7
            nsigma: Sigma-clipping limit for creating source mask, since we don't want to fit
                in very bright areas. Defaults to 5
            overwrite: Whether to overwrite or not. Defaults to False
        """

        if method not in ALLOWED_METHODS:
            raise ValueError(f"method should be one of {ALLOWED_METHODS}")

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.plot_dir = os.path.join(
            self.out_dir,
            "plots",
        )

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.method = method
        self.npixels = npixels
        self.separation = separation
        self.psf_fov_pixels = psf_fov_pixels
        self.psf_thresh = psf_thresh
        self.dilate_size = dilate_size
        self.nsigma = nsigma
        self.overwrite = overwrite

    def do_step(self):
        """Run PSF modelling"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "psf_model_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        files = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_{self.step_ext}.fits",
            )
        )
        files.sort()

        # Get a catalogue for the saturated coordinates
        sat_coords = self.get_sat_coords(files=files)

        if len(sat_coords) == 0:
            log.info(f"Found no saturated regions")
        else:
            log.info(f"Found {len(sat_coords)} saturated region(s) with coordinates:")
            for sat_coord in sat_coords:
                sat_coord_string = sat_coord.to_string(style="hmsdms", precision=2)
                log.info(f"-> {sat_coord_string}")

        # Feed these into the fitting routines
        success = self.run_step(
            files=files,
            sat_coords=sat_coords,
        )

        # If not everything has succeeded, then return a warning
        if not np.all(success) or len(success) != len(files):
            log.warning("Failures detected in PSF modelling")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
            self,
            files,
            sat_coords,
    ):
        """Run the step, fitting PSFs to catalogue positions of saturated coordinates

        Args:
            files: List of files to fit PSF for
            sat_coords: List of coordinates corresponding to saturated positions
        """

        success = []

        for file in files:

            file_short = os.path.split(file)[-1]
            log.info(f"Starting fit for {file_short}")

            file_out = os.path.join(self.out_dir,
                                    file_short,
                                    )

            with datamodels.open(file) as im:

                # If we don't have anything to mask, just save and continue
                if len(sat_coords) == 0:
                    im.save(file_out)
                    del im
                    success.append(True)
                    continue

                # Mask data we don't want to include
                dq_bit_mask = get_dq_bit_mask(
                    im.dq,
                )
                sat_mask = get_sat_mask(
                    im.dq,
                )

                data_masked = copy.deepcopy(im.data)
                data_masked[dq_bit_mask == 1] = np.nan
                err_masked = copy.deepcopy(im.err)
                err_masked[dq_bit_mask == 1] = np.nan

                # Get an array to put all the PSF models into
                full_psf_model = np.zeros_like(im.data)

                # Get PSF and shape
                psf = self.get_psf(file)
                psf_y_cen, psf_x_cen = centroid_com(psf)

                # Get average background level as an offset
                offset = sigma_clipped_stats(
                    im.data,
                    mask=dq_bit_mask,
                    maxiters=None,
                )[1]

                # We want a source mask, so we primarily fit to the low brightness outskirts
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mask = make_source_mask(
                        data_masked,
                        dilate_size=self.dilate_size,
                        nsigma=self.nsigma,
                    )

                for sat_coord in sat_coords:
                    sat_coord_string = sat_coord.to_string(style="hmsdms", precision=2)
                    log.info(f"Fitting for saturated region at {sat_coord_string}")

                    # Convert the RA/Dec into x/y
                    x_cen, y_cen = im.meta.wcs.invert(sat_coord.ra, sat_coord.dec)

                    # Get initial guess of the amplitude
                    init_amp = self.get_initial_amp(
                        data=data_masked - offset,
                        psf=psf,
                        x_cen=x_cen,
                        y_cen=y_cen,
                        psf_x_cen=psf_x_cen,
                        psf_y_cen=psf_y_cen,
                    )

                    if np.isnan(init_amp):
                        log.warning("Initial amplitude is NaN! Will skip fitting")
                        continue

                    pars = Parameters()

                    pars.add(
                        "amp",
                        value=init_amp,
                        min=0.1 * init_amp,
                        max=10 * init_amp,
                    )
                    pars.add(
                        "offset",
                        value=offset,
                        # vary=False,
                    )

                    # Here, x_cen and y_cen are offsets around 0, since we've already shuffled
                    pars.add(
                        "x_cen",
                        value=x_cen,
                        min=x_cen - 5,
                        max=x_cen + 5,
                    )
                    pars.add(
                        "y_cen",
                        value=y_cen,
                        min=y_cen - 5,
                        max=y_cen + 5,
                    )

                    result = minimize(
                        image_resid,
                        pars,
                        args=(
                            data_masked,
                            err_masked,
                            psf,
                            mask,
                            x_cen,
                            y_cen,
                            self.psf_thresh,
                        ),
                    )

                    log.info("Fit complete! Fit report:")
                    log.info(fit_report(result))

                    # Pull out best fit parameters and get this into the full PSF model
                    x_fit = result.params["x_cen"].value
                    y_fit = result.params["y_cen"].value
                    amp_fit = result.params["amp"].value
                    offset_fit = result.params["offset"].value

                    psf_model = np.zeros_like(im.data)
                    psf_model[: psf.shape[0], : psf.shape[1]] = copy.deepcopy(psf)
                    psf_model = shift(psf_model, [y_fit - psf_y_cen, x_fit - psf_x_cen])
                    full_psf_model += psf_model
                    full_psf_model *= amp_fit

                plot_name = os.path.join(self.plot_dir,
                                         file_short.replace(".fits", "")
                                         )

                if self.method == "replace":
                    mask = None
                elif self.method == "subtract":
                    mask = dq_bit_mask

                plot_success = self.make_diagnostic_plot(data=data_masked,
                                                         psf_model=full_psf_model + offset_fit,
                                                         plot_name=plot_name,
                                                         mask=mask,
                                                         )
                if not plot_success:
                    raise Warning(f"Issue with diagnostic plot for {file_short}")

                # Finally, either replace or subtract
                if self.method == "replace":

                    # Here, replace the saturated pixels and alter the DQ array appropriately
                    im.data[sat_mask] = full_psf_model[sat_mask] + offset_fit
                    im.dq[sat_mask] = 0

                elif self.method == "subtract":

                    # Here, subtract the full PSF model from the whole data array, but maintain overall
                    # flux level
                    im.data -= full_psf_model

                im.save(file_out)
                del im

                success.append(True)

            gc.collect()

        return success

    def get_sat_coords(self, files):
        """Get RA/Dec for the centres of saturated sources in each image

        Will look for saturated pixels in each image, and then merge these given a
        separation to a minimum catalogue

        Args:
            files: List of input files to loop over
        """

        log.info("Creating catalogue of saturated regions")

        # Code to get saturated clumps
        sat_coords = []

        # Get a mask of saturated pixels
        for input_file in files:
            with datamodels.open(input_file) as im:
                wcs = im.meta.wcs.to_fits_sip()
                w = WCS(wcs)

                # Get saturation mask
                sat_mask = get_sat_mask(im.dq)

                # Create a segmentation image
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    segment_map = detect_sources(
                        sat_mask,
                        threshold=0,
                        npixels=self.npixels,
                    )

                if segment_map is None:
                    continue

                # Go through this segmentation image and centroid each source
                for label in segment_map.labels:
                    label_map = segment_map.data == label
                    label_centroid = centroid_com(label_map)

                    ra, dec = w.all_pix2world(label_centroid[0], label_centroid[1], 0)
                    s = SkyCoord(ra * u.deg, dec * u.deg)

                    if len(sat_coords) > 0:
                        new_coord_found = True

                        for sat_coord in sat_coords:
                            if new_coord_found:
                                sep = s.separation(sat_coord)
                                if sep < self.separation * u.arcsec:
                                    new_coord_found = False

                        if new_coord_found:
                            sat_coords.append(s)

                    else:
                        sat_coords.append(s)

            del im

        return sat_coords

    def get_psf(
            self,
            file,
    ):
        """Get PSF for given observation

        Args:
            file: Input file to get PSF for
        """

        log.info("Generating PSF")

        inst = webbpsf.setup_sim_to_match_file(
            file,
            verbose=False,
        )
        inst.options["output_mode"] = "detector sampled"

        psf = inst.calc_psf(fov_pixels=self.psf_fov_pixels)

        # Pull out the data we care about
        psf_data = copy.deepcopy(psf["DET_DIST"].data)

        # Normalise to peak of 1
        psf_data /= np.nanmax(psf_data)

        return psf_data

    def get_initial_amp(
            self,
            data,
            psf,
            x_cen,
            y_cen,
            psf_x_cen,
            psf_y_cen,
    ):
        """Get initial amplitude guess for PSF

        This calculates an average ratio between the image and the PSF at
        the initial guess of the PSF centre. Our bounds for the amplitude are
        quite broad, so as long as this is order-of-magnitude right, we should be
        OK

        Args:
            data: Input data
            psf: Input PSF
            x_cen: Guess for x centre of saturated source
            y_cen: Guess for y centre of saturated source
            psf_x_cen: x centre of the PSF
            psf_y_cen: y centre of the PSF
        """

        psf_model = np.zeros_like(data)
        psf_model[: psf.shape[0], : psf.shape[1]] = copy.deepcopy(psf)
        psf_model = shift(psf_model, [y_cen - psf_y_cen, x_cen - psf_x_cen])

        # We want to isolate the region where the PSF is at least a little important
        psf_model[psf_model < self.psf_thresh * np.nanmax(psf)] = np.nan

        ratio = data / psf_model
        ratio[ratio == 0] = np.nan
        ratio[~np.isfinite(ratio)] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_amp_guess = sigma_clipped_stats(
                ratio,
                maxiters=None,
            )[1]

        return init_amp_guess

    def make_diagnostic_plot(self,
                             data,
                             psf_model,
                             plot_name,
                             mask=None,
                             ):
        """Create a diagnostic plot to show the fit

        If subtracting, will create data/fit PSF/subtracted data, otherwise
        will show data/fit PSF

        Args:
            data: Input data
            psf_model: Final PSF model
            plot_name: Name to save plot to
            mask: If not None, will NaN out pixels. Can make the visualisation
                clearer. Defaults to None
        """

        data = copy.deepcopy(data)
        psf_model = copy.deepcopy(psf_model)

        if mask is not None:
            psf_model[mask == 1] = np.nan

        if self.method == "replace":
            n_subplots = 2
        elif self.method == "subtract":
            n_subplots = 3
        else:
            raise ValueError(f"method should be one of {ALLOWED_METHODS}")

        vmin, vmax = np.nanpercentile(data, [1, 99])

        plt.figure(figsize=(4 * n_subplots, 5))
        ax1 = plt.subplot(1, n_subplots, 1)
        plt.imshow(
            data,
            origin="lower",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xticks([])
        plt.yticks([])

        plt.text(
            0.05,
            0.95,
            "Orig. data",
            ha="left",
            va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
            transform=ax1.transAxes,
        )

        ax2 = plt.subplot(1, n_subplots, 2, sharex=ax1, sharey=ax1)
        plt.imshow(
            psf_model,
            origin="lower",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xticks(visible=False)
        plt.yticks(visible=False)

        plt.text(
            0.05,
            0.95,
            "PSF model",
            ha="left",
            va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
            transform=ax2.transAxes,
        )

        if self.method == "subtract":
            ax3 = plt.subplot(1, n_subplots, 3, sharex=ax1, sharey=ax1)
            plt.imshow(
                data - psf_model,
                interpolation="none",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

            plt.xticks(visible=False)
            plt.yticks(visible=False)

            plt.text(
                0.05,
                0.95,
                "Subtracted",
                ha="left",
                va="top",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
                transform=ax3.transAxes,
            )

        plt.subplots_adjust(hspace=0, wspace=0)

        plt.savefig(f"{plot_name}.png", bbox_inches='tight')
        plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
        plt.close()

        return True
