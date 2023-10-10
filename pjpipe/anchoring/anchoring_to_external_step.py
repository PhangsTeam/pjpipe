import gc
import glob
import logging
import multiprocessing as mp
import os
import re
import shutil
import warnings
from functools import partial

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm
from ..psf_matching import do_jwst_convolution

from astropy.table import Table
from astropy.stats import mad_std
from collections import OrderedDict
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def line_func_curvefit(x, b, m):
    return b + m * x


def iterate_ols(x, y, e_y=None, guess=None,
                x0=None, s2nclip=3., iters=3):
    """
    Fit line to the data with iterative outliers and low S/N rejection
    """
    if guess is None:
        guess = [0., 1.]
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

    use = np.isfinite(x)
    for ii in range(iters):
        if s2nclip is None:
            if ii > 0:
                continue

        popt, pcurve = curve_fit(
            line_func_curvefit, x[use], y[use],
            sigma=e_y[use], p0=guess)

        intercept, slope = popt
        resid = y[use] - (intercept + slope * x[use])
        rms = mad_std(resid)

        if s2nclip is not None:
            use[use] = np.abs(resid) < (s2nclip * rms)

    return slope, intercept, rms


def bin_data(x, y, xmin=None, xmax=None, bin_step=None):
    """
    Calculate statistics in bins.
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
        bin_ind = (x >= this_xlo) * \
                  (x < this_xhi)

        this_y = y[bin_ind]

        stat_dict = OrderedDict()
        fin_ind = np.isfinite(this_y)
        for v in [16, 50, 84]:
            if np.sum(fin_ind) == 0:
                stat_dict[str(v)] = np.nan
            else:
                stat_dict[str(v)] = np.percentile(this_y[fin_ind], v)

        stat_dict['xmid'] = this_xctr
        stat_dict['xlo'] = this_xlo
        stat_dict['xhi'] = this_xhi
        stats_for_bins.append(stat_dict)

    bin_table = Table(stats_for_bins)

    return bin_table


def solve_for_offset(comp_data, ref_data, mask=None,
                     xmin=0.25, xmax=3.5, binsize=0.1,
                     save_plot=None, label_str='Comparison'):
    """Solve for the offset between two images, optionally also allowing
    for a free slope relating them and restricting to a specific range
    of values or applying an extra spatial mask.

    Inputs:



    Returns:

    offset, slope : the value to subtract from COMP to match the zero
    point of REF assuming they can be related by a single scaling,
    along with the SLOPE to multiply REF by to get COMP after removing
    a DC offset. That is:

    comp = slope * ref + offset

    """

    # Identify overlap used to solve for the offset

    overlap = np.isfinite(comp_data) * np.isfinite(ref_data)

    if mask is not None:
        overlap = overlap * mask

    # Solve for the difference and note statistics
    comp_vec = comp_data[overlap]
    ref_vec = ref_data[overlap]

    comp_bins = bin_data(ref_vec, comp_vec, xmin=xmin, xmax=xmax, bin_step=binsize)
    xbins = comp_bins['xmid']
    ybins = comp_bins['50']

    slope, intercept, resid = iterate_ols(xbins, ybins, e_y=None, x0=None, s2nclip=3., iters=3, guess=[0.0, 1.0])

    # Optionally make diagnostic plots
    if save_plot:

        xlim_lo = -1.
        xlim_hi = 5.0

        fig, ax = plt.subplots()

        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(xlim_lo, xlim_hi)
        ax.grid(True, linestyle='dotted', linewidth=0.5, color='black', zorder=2)

        ax.scatter(ref_vec, comp_vec, marker='.', color='gray', s=1, zorder=1)

        xbins = comp_bins['xmid']
        ybins = comp_bins['50']
        lo_ybins = comp_bins['16']
        hi_ybins = comp_bins['84']

        ax.scatter(comp_bins['xmid'], comp_bins['50'], color='red', marker='o', s=50, zorder=5)
        ax.errorbar(xbins, ybins, [(ybins - lo_ybins), (hi_ybins - ybins)],
                    color='red', capsize=0.1, elinewidth=2, fmt='none',
                    zorder=4)

        fidx = np.arange(xlim_lo, xlim_hi, 0.01)
        ax.plot(fidx, fidx * slope + intercept, linewidth=3,
                color='black', zorder=6, alpha=0.5, linestyle='dashed')

        bbox_props = dict(boxstyle="round", fc="lightgray", ec='black', alpha=0.9)
        yval = 0.95
        va = 'top'

        this_label = label_str + '\n' + 'm = ' + str(slope) + '\n' + 'b=' + str(intercept)
        ax.text(0.04, yval, this_label,
                ha='left', va=va,
                transform=ax.transAxes,
                size='small', bbox=bbox_props,
                zorder=5)
        plt.savefig(save_plot, bbox_inches='tight')

    return slope, intercept


class AnchoringStep:
    def __init__(
        self,
        target,
        w_dir,
        reference_dir,
        procs,
        step_ext_in,
        step_ext_out='anc',
        subdir_in=None,
        subdir_out=None,
        kernels_dir=None,
        overwrite=False,
        reference=None,
        ref_band=None,
        external_band=None,
        all_bands=None

    ):
        """Anchor aligned data to the external images"""

        self.target = target
        self.w_dir = w_dir
        self.subdir_in = subdir_in
        self.subdir_out = subdir_out
        self.procs = procs
        self.step_ext_in = step_ext_in
        self.step_ext_out = step_ext_out
        self.overwrite = overwrite
        self.kernels_dir = kernels_dir
        if reference is None or not reference.get(target):
            self.reference = None
        self.reference = {'nircam': os.path.join(reference_dir, reference['nircam'].get(target)),
                          'miri': os.path.join(reference_dir, reference['miri'].get(target)),
                          }
        self.ref_band = ref_band
        self.all_bands = all_bands
        self.external_band = external_band

    def do_step(self):
        """Run anchoring step"""

        if (self.reference is None
                or not (os.path.exists(self.reference['nircam']) and os.path.exists(self.reference['miri']))):
            log.error("Cannot find reference image for anchoring step. Skip this step.")
            return False

        if (self.kernels_dir is None or not os.path.exists(self.kernels_dir)) and self.subdir_out is None:
            log.error("Either kernels or path to the psf-matched images should be provided for anchoring step. "
                      "Skip this step")
            return False

        step_complete_file = os.path.join(
            self.w_dir,
            "anchoring_to_external_step_complete.txt",
        )

        if self.overwrite:
            for band in self.all_bands:
                files_to_remove = glob.glob(
                    os.path.join(
                        self.w_dir, band.upper(), self.subdir_out,
                        f"*_{self.step_ext_out}.fits",
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
        for band in self.all_bands:
            cur_files = glob.glob(os.path.join(self.w_dir, band.upper(), self.subdir_in, f"*_{self.step_ext_in}.fits"))
            files.extend(cur_files)
        files.sort()

        successes = self.run_step(
            files,
        )

        if not np.all(successes):
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

        Args:
            files: List of files to process
        """

        log.info(f"Applying anchoring to external images")

        files_for_external = [f for f in files
                              if self.ref_band['nircam'].lower() in os.path.split(f)[-1]]
        files_for_external.extend([f for f in files if self.ref_band['miri'].lower() in os.path.split(f)[-1]])

        if len(files_for_external) == 0:
            log.warning("Cannot proceed as files for comparison with external images are not found. Skip step")
            return [False]*len(files)

        files_for_internal = {'nircam': [f for f in files
                                         if self.ref_band['nircam'].lower() not in os.path.split(f)[-1]
                                         and 'nircam' in os.path.split(f)[-1]],
                              'miri': [f for f in files
                                       if self.ref_band['miri'].lower() not in os.path.split(f)[-1]
                                       and 'miri' in os.path.split(f)[-1]]
                              }

        offsets = []
        for band in self.all_bands:
            if not os.path.exists(os.path.join(self.w_dir, band.upper(), self.subdir_out)):
                os.makedirs(os.path.join(self.w_dir, band.upper(), self.subdir_out))
        # == First, anchor selected bands to external images (NIRCAM vs External and MIRI vs External)
        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files_for_external)])
        with mp.get_context("fork").Pool(procs) as pool:

            for offset in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_anchoring,
                        external=True,
                        internal_reference=None,
                        ref_offset=0.
                    ),
                    files_for_external,
                ),
                ascii=True,
                desc="Applying anchoring to external images",
                total=len(files_for_external),
            ):
                offsets.append(offset)

            pool.close()
            pool.join()
            gc.collect()

        ref_offsets = {'nircam': offsets[0], 'miri': offsets[1]}

        # == Next, apply internal anchoring (NIRCAM vs NIRCAM and MIRI vs MIRI)
        for instrument in ['nircam', 'miri']:
            internal_reference = [f for f in files_for_external
                                  if instrument in os.path.split(f)[-1]][0]
            procs = np.nanmin([self.procs, len(files_for_internal[instrument])])
            with mp.get_context("fork").Pool(procs) as pool:
                for offset in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_anchoring,
                            external=False,
                            internal_reference=internal_reference,
                            ref_offset=ref_offsets[instrument]
                        ),
                        files_for_internal[instrument],
                    ),
                    ascii=True,
                    desc=f"Applying internal anchoring to {instrument} images",
                    total=len(files_for_internal[instrument]),
                ):
                    offsets.append(offset)

                pool.close()
                pool.join()
                gc.collect()

        successes = np.isfinite(np.array(offsets))
        return successes

    def parallel_anchoring(
        self,
        file,
        external=True,
        internal_reference=None,
        ref_offset=0.0,
    ):
        """Parallelize applying anchoring to external images

        Args:
            file: File to apply anchoring
            external: anchoring to external (True) or internal (False) images
            internal_reference: path to the internal reference image (None if external = True)
            ref_offset: offset calculated for the reference image (should be 0 if external = True)

        Returns:
            offset value
        """

        file_short = os.path.split(file)[-1]
        current_band = "".join(re.findall("(f\d+[mwn])", file_short))
        file_short = file_short.replace(self.step_ext_in, self.step_ext_out)

        output_file = os.path.join(self.w_dir, current_band.upper(), self.subdir_out, file_short)
        if 'nircam' in file_short:
            instrument = 'nircam'
        elif 'miri' in file_short:
            instrument = 'miri'
        else:
            log.error('Unrecognized camera during the anchoring')
            return np.nan

        if external:
            ref_file = self.reference[instrument]
            ref_band = self.external_band[instrument]
            if self.target.lower() in ref_band:
                ref_band = ref_band[self.target.lower()]
            else:
                ref_band = ref_band['default']
            conv_band = ref_band
        else:
            ref_file = internal_reference
            ref_band = self.ref_band[instrument]
            conv_band = 'F2100W'

        # === First, convolve image to the target resolution and save it in the matched_dir
        # (or use those already available there)
        if ref_band == conv_band:
            ref_file_conv = ref_file
        else:
            ref_file_short = os.path.split(ref_file)[-1].split('.fits')[0]
            ref_file_conv = os.path.join(self.w_dir, ref_band.upper(), self.subdir_out,
                                         f"{ref_file_short}_at{conv_band}.fits")

        if current_band.upper() == conv_band.upper():
            file_conv = file
        else:
            file_short = os.path.split(file)[-1].split('.fits')[0]
            file_conv = os.path.join(self.w_dir, current_band.upper(), self.subdir_out,
                                     f"{file_short}_at{conv_band}.fits")

            if not os.path.exists(file_conv):
                kernel_file = os.path.join(self.kernels_dir, f"{current_band.upper()}_to_{conv_band.upper()}.fits")
                if not os.path.exists(kernel_file):
                    log.error('Cannot convolve file to compare with reference as the kernel does not exist')
                    return np.nan
                do_jwst_convolution(file, file_conv, kernel_file)
            if external:
                # convolve internal reference image also to F2100W as we will need it on the next iteration
                kernel_file = os.path.join(self.kernels_dir, f"{current_band.upper()}_to_F2100W.fits")
                f_out = os.path.join(self.w_dir, current_band.upper(), self.subdir_out,
                                     f"{file_short}_atF2100W.fits")
                if not os.path.exists(f_out):
                    do_jwst_convolution(file, f_out, kernel_file)

        # === Reproject current image to the ref_image wcs
        image, header = fits.getdata(file_conv, header=True, extname='SCI')
        if external:
            # Assume that science data is in primary extension for external images
            image_ref, header_ref = fits.getdata(ref_file_conv, header=True, ext=0)
        else:
            image_ref, header_ref = fits.getdata(ref_file_conv, header=True, extname='SCI')

        repr_image, fp = reproject_interp((image, header),
                                          output_projection=WCS(header_ref), shape_out=image_ref.shape)
        fp = fp.astype(bool)
        repr_image[~fp] = np.nan

        # === Calculate intercept (and slope) between image and reference

        if ref_band == 'W3':
            xmin = 0.0
            xmax = 0.8
        else:
            xmin = 0.25
            xmax = 2.0
        saveplot_filename = os.path.join(self.w_dir, current_band.upper(), self.subdir_out,
                                         f"{self.target}_{current_band}_vs_{ref_band}_compar.png")
        slope, intercept = solve_for_offset(repr_image, image_ref, xmin=xmin, xmax=xmax, binsize=0.1,
                                            save_plot=saveplot_filename,
                                            label_str=self.target.upper() + '\n' +
                                                      current_band.upper() + ' vs. ' + ref_band.upper())

        offset = -intercept + slope * ref_offset

        # === Save anchored file
        with fits.open(file) as hdu:
            hdu['SCI'].data[np.isfinite(hdu['SCI'].data) & (hdu['SCI'].data != 0)] += offset
            hdu[0].header['BKGRDVAL'] = offset
            hdu.writeto(output_file, overwrite=True)

        return offset




