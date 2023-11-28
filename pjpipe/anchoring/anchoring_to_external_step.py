import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import re
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from reproject import reproject_interp
from scipy.optimize import curve_fit
from tqdm import tqdm

from ..utils import do_jwst_convolution, get_band_type

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def line_func_curvefit(x, b, m):
    """Just an equation for a straight line"""
    return b + m * x


def iterate_ols(x, y, e_y=None, guess=None, x0=None, s2nclip=3.0, iters=3):
    """
    Fit line to data with iterative outliers and low S/N rejection

    Args:
        x: x-coords
        y: y-coords
        e_y: y-errors. Defaults to None, i.e. no error in the measurements
        guess: Initial guess for fit parameters. Defaults to None, which will be [0, 1]
        x0: Offset to subtract from x values. Defaults to None (i.e. no subtraction)
        s2nclip: Clip value for sigma-clipping. Defaults to 3
        iters: Number of iterations to run. Defaults to 3
    """
    if guess is None:
        guess = [0.0, 1.0]
    if x0 is not None:
        x = x - x0

    if e_y is None:
        e_y = 1.0 + y * 0.0
    elif type(e_y) == type(1.0):
        e_y = e_y + y * 0.0
    elif len(e_y) == 1:
        e_y = e_y + y * 0.0

    fin_ind = np.isfinite(x) * np.isfinite(y) * np.isfinite(e_y)
    x = x[fin_ind]
    y = y[fin_ind]
    e_y = e_y[fin_ind]

    slope, intercept, rms = None, None, None

    use = np.isfinite(x)
    for ii in range(iters):
        if s2nclip is None:
            if ii > 0:
                continue

        # Old version didn't clip properly but used MAD. This can crash,
        # so use std instead
        # popt, pcurve = curve_fit(
        #     line_func_curvefit, x, y, sigma=e_y, p0=guess
        # )
        popt, pcurve = curve_fit(
            line_func_curvefit, x[use], y[use], sigma=e_y[use], p0=guess
        )

        intercept, slope = popt
        resid = y[use] - (intercept + slope * x[use])
        # rms = mad_std(resid)
        rms = np.nanstd(resid)

        if s2nclip is not None:
            use[use] = np.abs(resid) < (s2nclip * rms)

    if slope is None:
        raise ValueError("Something has gone wrong in the fitting!")

    return slope, intercept, rms


def bin_data(x, y, xmin=None, xmax=None, bin_step=None):
    """
    Calculate statistics in bins.

    Args:
        x: x values
        y: y values
        xmin: Minimum value for binning
        xmax: Maximum value for binning
        bin_step: Step in x for the binning
    """

    # === Calculate x bin edges
    half_width = bin_step / 2.0
    extent = xmax - xmin

    # Subtract 0.5 width from each side then calculate number of
    # steps, either ensuring totally-within-range or
    # covers-whole-range according to keywords.

    nbin = int(np.ceil((extent - bin_step) / bin_step))

    bin_ind = np.arange(0, nbin, 1)
    xmid = bin_ind * bin_step + (xmin + bin_step * 0.5)

    xlo = xmid - half_width
    xhi = xmid + half_width

    # Identify the measurements we work with
    fin_ind = np.isfinite(x) * np.isfinite(y)
    x = x[fin_ind]
    y = y[fin_ind]

    # Initialize the output
    stats_for_bins = []

    # Loop and calculate statistics in each bin
    for ii, low_edge in enumerate(xlo):
        # note boundaries
        this_xlo = xlo[ii]
        this_xhi = xhi[ii]
        this_xctr = (this_xlo + this_xhi) * 0.5

        # for this bin get the data ...
        bin_ind = (x >= this_xlo) * (x < this_xhi)

        this_y = y[bin_ind]

        stat_dict = OrderedDict()
        fin_ind = np.isfinite(this_y)
        for v in [16, 50, 84]:
            if np.sum(fin_ind) == 0:
                stat_dict[str(v)] = np.nan
            else:
                stat_dict[str(v)] = np.percentile(this_y[fin_ind], v)

        stat_dict["xmid"] = this_xctr
        stat_dict["xlo"] = this_xlo
        stat_dict["xhi"] = this_xhi
        stats_for_bins.append(stat_dict)

    bin_table = Table(stats_for_bins)

    return bin_table


def solve_for_offset(
    comp_data,
    ref_data,
    mask=None,
    xmin=0.25,
    xmax=3.5,
    binsize=0.1,
    xlim_plot=None,
    save_plot=None,
    label_str="Comparison",
):
    """Solve for the offset between two images

    This optionally also allowing for a free slope relating them and restricting
    to a specific range of values or applying an extra spatial mask.

    offset, slope : the value to subtract from COMP to match the zero
    point of REF assuming they can be related by a single scaling,
    along with the SLOPE to multiply REF by to get COMP after removing
    a DC offset. That is:

    comp = slope * ref + offset

    Args:
        comp_data: Comparison data
        ref_data: Reference data
        mask: Optional mask. Defaults to None (no mask)
        xmin: Minimum x-value for binning. Defaults to 0.25
        xmax: Max x-value for binning. Defaults to 3.5
        binsize: Size of the bins. Defaults to 0.1
        xlim_plot: Plot limits to avoid outliers. Defaults to None
        save_plot: If a string, will safe to that file. Defaults to None
        label_str: String to put in text label. Defaults to 'comparison'
    """

    # Identify overlap used to solve for the offset

    if xlim_plot is None:
        xlim_plot = [-1, 5]

    overlap = np.isfinite(comp_data) * np.isfinite(ref_data)

    if mask is not None:
        overlap = overlap * mask

    # Solve for the difference and note statistics
    comp_vec = comp_data[overlap]
    ref_vec = ref_data[overlap]

    comp_bins = bin_data(ref_vec, comp_vec, xmin=xmin, xmax=xmax, bin_step=binsize)
    xbins = comp_bins["xmid"]
    ybins = comp_bins["50"]

    slope, intercept, resid = iterate_ols(
        xbins, ybins, e_y=None, s2nclip=3.0, iters=3, guess=[0.0, 1.0]
    )

    # Optionally make diagnostic plots
    if save_plot:
        fig, ax = plt.subplots()

        ax.set_xlim(xlim_plot)
        ax.set_ylim(xlim_plot)
        ax.grid(True, linestyle="dotted", linewidth=0.5, color="black", zorder=2)

        ax.scatter(ref_vec, comp_vec, marker=".", color="gray", s=1, zorder=1)

        xbins = comp_bins["xmid"]
        ybins = comp_bins["50"]
        lo_ybins = comp_bins["16"]
        hi_ybins = comp_bins["84"]

        ax.scatter(
            comp_bins["xmid"], comp_bins["50"], color="red", marker="o", s=50, zorder=5
        )
        ax.errorbar(
            xbins,
            ybins,
            [(ybins - lo_ybins), (hi_ybins - ybins)],
            color="red",
            capsize=0.1,
            elinewidth=2,
            fmt="none",
            zorder=4,
        )

        fidx = np.arange(*xlim_plot, 0.01)
        ax.plot(
            fidx,
            fidx * slope + intercept,
            linewidth=3,
            color="black",
            zorder=6,
            alpha=0.5,
            linestyle="dashed",
        )

        bbox_props = dict(boxstyle="round", fc="lightgray", ec="black", alpha=0.9)
        yval = 0.95
        va = "top"

        this_label = f"{label_str}\n m={str(slope)} \n b={str(intercept)}"
        ax.text(
            0.04,
            yval,
            this_label,
            ha="left",
            va=va,
            transform=ax.transAxes,
            size="small",
            bbox=bbox_props,
            zorder=5,
        )
        plt.savefig(f"{save_plot}.png", bbox_inches="tight")
        plt.savefig(f"{save_plot}.pdf", bbox_inches="tight")

    return slope, intercept


class AnchoringStep:
    def __init__(
        self,
        target,
        bands,
        in_dir,
        ref_dir,
        procs,
        in_step_ext,
        out_step_ext="i2d_anchor",
        in_subdir=None,
        out_subdir=None,
        kernel_dir=None,
        ref_band=None,
        external_bands=None,
        internal_conv_band=None,
        overwrite=False,
    ):
        """Anchor aligned data to the external images

        Will convolve data to a common resolution given a reference image, bin and fit
        for an offset between them. Can be used with external images to anchor to known fluxes,
        or JWST images for internal consistency.

        N.B. for external files the naming scheme expects something like [target]_[resolution].fits,
        where resolution can be something like 'irac1', or if pre-convolved, then maybe 'irac1_atgauss15'.
        Kernels should be named something like [resolution]_to_[resolution].fits. This is non-negotiable!

        Args:
            target: Target to consider
            bands: Bands to consider
            in_dir: Input directory
            ref_dir: Directory for reference images
            procs: Number of processes to run in parallel
            in_step_ext: Filename extension for the input files
            out_step_ext: Filename extension for output files. Defaults to "i2d_anchor"
            in_subdir: Where files are located within the target directory (something like 'lv3')
            out_subdir: Where to save intermediate files (something like 'anchored')
            kernel_dir: Where kernels are located
            ref_band: If internally anchoring, these are the bands to lock to
            external_bands: If externally anchoring, these are a list (in preference order) of resolutions
                to lock to (e.g. irac1, irac1_atgauss4p5)
            internal_conv_band: For internal anchoring, we use this common resolution.
            overwrite: Whether to overwrite or not
        """

        if kernel_dir is None or not os.path.exists(kernel_dir):
            raise ValueError("kernel_dir should be defined and should exist")
        if out_subdir is None:
            raise ValueError("out_subdir should be defined")
        if external_bands is None:
            raise ValueError("external_bands should be defined")
        if internal_conv_band is None:
            raise ValueError("internal_conv_band should be defined")

        # Make sure we're definitely looping over a list here
        if isinstance(external_bands, str):
            external_bands = [external_bands]
        if isinstance(external_bands, dict):
            for key in external_bands.keys():
                if isinstance(external_bands[key], str):
                    external_bands[key] = [external_bands[key]]

        self.target = target
        self.bands = bands
        self.in_dir = in_dir
        self.ref_dir = ref_dir
        self.procs = procs
        self.in_step_ext = in_step_ext
        self.out_step_ext = out_step_ext
        self.in_subdir = in_subdir
        self.out_subdir = out_subdir
        self.kernel_dir = kernel_dir
        self.ref_band = ref_band
        self.external_bands = external_bands
        self.internal_conv_band = internal_conv_band
        self.overwrite = overwrite

    def do_step(self):
        """Run anchoring step"""

        step_complete_file = os.path.join(
            self.in_dir,
            "anchoring_to_external_step_complete.txt",
        )

        if self.overwrite:
            for band in self.bands:
                files_to_remove = glob.glob(
                    os.path.join(
                        self.in_dir,
                        band.upper(),
                        self.out_subdir,
                        f"*_{self.out_step_ext}.fits",
                    )
                )
                for fname in files_to_remove:
                    os.remove(fname)
                if os.path.exists(step_complete_file):
                    os.remove(step_complete_file)

        # Check if we've already run the step
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        files = []
        for band in self.bands:
            cur_files = glob.glob(
                os.path.join(
                    self.in_dir,
                    band,
                    self.in_subdir,
                    f"*_{self.in_step_ext}.fits",
                )
            )
            files.extend(cur_files)
        files.sort()

        success = self.run_step(
            files,
        )

        if not success:
            log.warning("Failures detected in applying anchoring to external images")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
        self,
        files,
    ):
        """Wrap paralellism around applying anchoring to external images

        This will loop over NIRCam/MIRI files (excluding backgrounds) and perform
        anchoring as set up by the user. Will put out a table of fit values and
        the anchored images at the end

        Args:
            files: List of files to process
        """

        log.info(f"Applying anchoring to external images")

        # Don't anchor background files
        files_for_external = {
            "nircam": [
                f
                for f in files
                if self.ref_band["nircam"].lower() in os.path.split(f)[-1]
                and "_bgr_" not in os.path.split(f)[-1]
            ],
            "miri": [
                f
                for f in files
                if self.ref_band["miri"].lower() in os.path.split(f)[-1]
                and "_bgr_" not in os.path.split(f)[-1]
            ],
        }

        if len(files_for_external) == 0:
            log.warning(
                "Cannot proceed as files for comparison with external images are not found. Skip step"
            )
            return [False] * len(files)

        files_for_internal = {
            "nircam": [
                f
                for f in files
                if self.ref_band["nircam"].lower() not in os.path.split(f)[-1]
                and "nircam" in os.path.split(f)[-1]
                and "_bgr_" not in os.path.split(f)[-1]
            ],
            "miri": [
                f
                for f in files
                if self.ref_band["miri"].lower() not in os.path.split(f)[-1]
                and "miri" in os.path.split(f)[-1]
                and "_bgr_" not in os.path.split(f)[-1]
            ],
        }

        for band in self.bands:
            out_dir = os.path.join(self.in_dir, band, self.out_subdir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # First, anchor selected bands to external images (NIRCAM vs External and MIRI vs External)
        anchor_tab = Table(
            names=[
                "target",
                "ref_filter",
                "compar_filename",
                "compar_filter",
                "slope",
                "intercept",
            ],
            dtype=[
                str,
                str,
                str,
                str,
                float,
                float,
            ],
        )
        offsets = {}

        # Pull out all the possible combos here
        files_for_external_w_ref = []
        for key in files_for_external:
            for f in files_for_external[key]:
                external_bands = self.external_bands[key]
                for external_band in external_bands:
                    files_for_external_w_ref.append([f, external_band])

        # Because we can have things overlapping, do this in serial rather than parallel
        offset_tab_rows = []
        for f in tqdm(
            files_for_external_w_ref,
            ascii=True,
            desc="Applying anchoring to external images",
            total=len(files_for_external_w_ref),
        ):
            offset = self.parallel_anchoring(
                f,
                external=True,
                internal_reference=None,
            )
            offset_tab_rows.append(offset)

        # Add these to the table
        for tab_row in offset_tab_rows:
            if tab_row is not None:
                anchor_tab.add_row(tab_row)

        ref_offsets = {}

        # Find a fiducial offset based on the priority of filters
        for band in np.unique(anchor_tab["compar_filter"]):
            found_offset = False
            offset = 0
            filename = None

            band_rows = anchor_tab[anchor_tab["compar_filter"] == band]
            band_type = get_band_type(band.upper())

            for external_band in self.external_bands[band_type]:
                ref_filter = f"{self.target}_{external_band}"

                if not found_offset:
                    external_band_row = band_rows[band_rows["ref_filter"] == ref_filter]
                    if len(external_band_row) == 1:
                        offset = -external_band_row["intercept"][0]
                        filename = external_band_row["compar_filename"][0]
                        found_offset = True

            if not found_offset:
                raise ValueError("Could not find a suitable reference offset")

            ref_offsets[band_type] = offset
            offsets[band] = {
                "filename": filename,
                "offset": offset,
            }

        offset_tab_rows = []

        # Next, apply internal anchoring (NIRCAM vs NIRCAM and MIRI vs MIRI)
        for instrument in ["nircam", "miri"]:
            try:
                internal_reference = [f for f in files_for_external[instrument]][0]
            except IndexError:
                log.warning(f"Internal reference for {instrument} not found. Skipping")
                continue

            if len(files_for_internal[instrument]) == 0:
                log.warning(
                    f"No internal files found to anchor for {instrument}. Skipping"
                )
                continue

            procs = np.nanmin([self.procs, len(files_for_internal[instrument])])
            with mp.get_context("fork").Pool(procs) as pool:
                for offset in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_anchoring,
                            external=False,
                            internal_reference=internal_reference,
                        ),
                        files_for_internal[instrument],
                    ),
                    ascii=True,
                    desc=f"Applying internal anchoring to {instrument} images",
                    total=len(files_for_internal[instrument]),
                ):
                    offset_tab_rows.append(offset)

                pool.close()
                pool.join()
                gc.collect()

        # Add these to the table
        for tab_row in offset_tab_rows:
            if tab_row is not None:
                anchor_tab.add_row(tab_row)

        # Sort the table
        anchor_tab.sort(keys="compar_filter")

        # Find offsets for the other bands
        for band in np.unique(anchor_tab["compar_filter"]):
            if band in offsets:
                continue

            row = anchor_tab[anchor_tab["compar_filter"] == band]
            filename = row["compar_filename"][0]

            band_type = get_band_type(band.upper())

            ref_offset = ref_offsets[band_type]
            offset = -row["intercept"][0] + row["slope"][0] * ref_offset

            ref_offsets[band_type] = offset
            offsets[band] = {
                "filename": filename,
                "offset": offset,
            }

        log.info("Writing out files")

        # Write out all the offset files
        for band in offsets:
            filename = offsets[band]["filename"]
            offset = offsets[band]["offset"]

            input_file = os.path.join(
                self.in_dir, band.upper(), self.in_subdir, f"{filename}.fits"
            )
            output_file = os.path.join(
                self.in_dir,
                band.upper(),
                self.in_subdir,
                f"{filename.replace(self.in_step_ext, self.out_step_ext)}.fits",
            )

            # Save anchored files
            with fits.open(input_file) as hdu:
                hdu["SCI"].data[
                    np.isfinite(hdu["SCI"].data) & (hdu["SCI"].data != 0)
                ] += offset
                hdu[0].header["BKGRDVAL"] = offset

                # TODO: We should also include an error term here. It's just a single number, but for internal
                #   will be a quadrature sum

                hdu.writeto(output_file, overwrite=True)

        log.info("Writing out anchor table")

        out_tab_name = os.path.join(self.in_dir, f"{self.target}_anchor_tab.fits")
        anchor_tab.write(out_tab_name, overwrite=True)

        return True

    def parallel_anchoring(
        self,
        file,
        external=True,
        internal_reference=None,
    ):
        """Parallelize applying anchoring to external images

        Does relevant convolutions and fits for the anchoring offsets

        Args:
            file: File and (potentially) reference to apply anchoring
            external: anchoring to external (True) or internal (False) images
            internal_reference: path to the internal reference image (None if external = True)

        """

        external_band = None
        if external:
            external_band = copy.deepcopy(file[1])
            file = copy.deepcopy(file[0])

        file_short = os.path.split(file)[-1].split(".fits")[0]

        current_band = "".join(re.findall("(f\d+[mwn])", file_short))

        instrument = file_short.split("_")[1]

        if external:
            ref_file = os.path.join(
                self.ref_dir,
                f"{self.target}_{external_band}.fits",
            )
            ref_file_short = os.path.split(ref_file)[-1]

            # If we've already convolved (e.g. there's an _at in there), parse that
            # bit. Otherwise, assume at native res and parse
            if "_at" in ref_file_short:
                ref_band = ref_file_short.split("_at")[-1]
            else:
                ref_band = ref_file_short.split("_")[-1]
            ref_band = ref_band.split(".fits")[0]
            conv_band = copy.deepcopy(ref_band)
        else:
            ref_file = internal_reference
            ref_file_short = os.path.split(ref_file)[-1]
            ref_band = self.ref_band[instrument]
            conv_band = copy.deepcopy(self.internal_conv_band)

        # Clean up the reference file
        ref_file_clean = ref_file_short.split(".fits")[0]

        # First, convolve image to the target resolution and save it in the matched_dir
        # (or use those already available there)
        if ref_band == conv_band:
            ref_file_conv = ref_file
        else:
            ref_file_conv = os.path.join(
                self.in_dir,
                ref_band.upper(),
                self.out_subdir,
                f"{ref_file_clean}_at{conv_band.lower()}.fits",
            )

        # If the reference file doesn't exist, return
        if not os.path.exists(ref_file_conv):
            return None

        # If we need to, do the convolutions
        if current_band.upper() == conv_band.upper():
            file_conv = file
        else:
            file_conv = os.path.join(
                self.in_dir,
                current_band.upper(),
                self.out_subdir,
                f"{file_short}_at{conv_band.lower()}.fits",
            )

            if not os.path.exists(file_conv):
                kernel_file = os.path.join(
                    self.kernel_dir,
                    f"{current_band.lower()}_to_{conv_band.lower()}.fits",
                )
                if not os.path.exists(kernel_file):
                    log.error(
                        "Cannot convolve file to compare with reference as the kernel does not exist"
                    )
                    return None
                do_jwst_convolution(file, file_conv, kernel_file)
            if external:
                # convolve internal reference image also to the internal convolution band as we will
                # need it on the next iteration
                kernel_file = os.path.join(
                    self.kernel_dir,
                    f"{current_band.lower()}_to_{self.internal_conv_band.lower()}.fits",
                )
                if not os.path.exists(kernel_file):
                    log.error(
                        "Cannot convolve file to internal convolution resolution as the kernel does not exist"
                    )
                    return None
                f_out = os.path.join(
                    self.in_dir,
                    current_band.upper(),
                    self.out_subdir,
                    f"{file_short}_at{self.internal_conv_band.lower()}.fits",
                )
                if not os.path.exists(f_out):
                    do_jwst_convolution(file, f_out, kernel_file)

        # Reproject current image to the ref_image wcs
        image, header = fits.getdata(file_conv, header=True, extname="SCI")
        if external:
            # Assume that science data is in primary extension for external images
            image_ref, header_ref = fits.getdata(ref_file_conv, header=True, ext=0)
        else:
            image_ref, header_ref = fits.getdata(
                ref_file_conv, header=True, extname="SCI"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            repr_image, fp = reproject_interp(
                (image, header),
                output_projection=WCS(header_ref),
                shape_out=image_ref.shape,
            )
        fp = fp.astype(bool)
        repr_image[~fp] = np.nan

        # Calculate intercept (and slope) between image and reference
        if "w3" in ref_file_clean:
            xmin = 0.0
            xmax = 0.8
        else:
            xmin = 0.25
            xmax = 2.0

        saveplot_filename = os.path.join(
            self.in_dir,
            current_band.upper(),
            self.out_subdir,
            f"{self.target}_{current_band}_vs_{ref_file_clean}_compar",
        )
        slope, intercept = solve_for_offset(
            repr_image,
            image_ref,
            xmin=xmin,
            xmax=xmax,
            binsize=0.1,
            save_plot=saveplot_filename,
            label_str=f"{self.target}\n{current_band} vs. {ref_file_clean}",
        )

        return self.target, ref_file_clean, file_short, current_band, slope, intercept
