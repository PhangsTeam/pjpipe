import copy
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from astropy import units as u
from astropy.time import Time
from jwst import datamodels
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from uncertainties import ufloat

matplotlib.use("agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14

log = logging.getLogger(__name__)

# From https://arxiv.org/pdf/2512.15477
CORRECTION_COEFFICIENTS = {
    "A": ufloat(1.85, 0.13),
    "tau": ufloat(-0.45, 0.10),
}


def apply_persistence_correction(
    in_file,
    out_file,
    corr_dict,
):
    """Apply persistence correction to an image

    Args:
        in_file: Input fits file
        out_file: Output fits file
        corr_dict: Dictionary of correction factors (in percent)
    """

    with datamodels.open(in_file) as im:

        pers_corr = np.zeros_like(im.data)

        for f_other, vals in corr_dict.items():
            if "corr_factor_val" in vals:
                # Apply the correction factor, remembering this is a percent
                # so needs a factor 100
                with datamodels.open(f_other) as im_other:
                    pers_corr_im = im_other.data * vals["corr_factor_val"] / 100

                    # Set any NaNs to 0s, as well as any negatives
                    pers_corr_im[
                        np.logical_or(~np.isfinite(pers_corr_im), pers_corr_im < 0)
                    ] = 0
                pers_corr += pers_corr_im

        # Subtract the persistence correction from the data
        im.data -= pers_corr
        im.save(out_file)

        del im

    return pers_corr


class PersistenceStep:
    def __init__(
        self,
        band,
        in_dir,
        out_dir,
        step_ext,
        procs,
        use_all_bands=False,
        apply_only_within_visits=False,
        correction_coefficients=None,
        overwrite=False,
    ):
        """Correct MIRI images for persistence

        Since the official pipeline does not yet have
        a working persistence correction, this is our
        ad-hoc solution. Looping over observations,
        we perform a correction using a time-dependent
        exponential decay model which is the same across
        all bands

        This correction takes the form:

        p = A * exp(tau * t)

        Where A is an amplitude, t is in hours, and tau the
        (negative) decay coefficient. This should yield
        a correction in terms of flux percentage, which will
        be subtracted from a frame.

        We also include options for using corrections from all
        bands, not just the band being corrected, as well
        as the option to only correct using observations
        within a particular visit.

            N.B. These have not yet been optimized! We do not
            have a recommendation

        Args:
            band: Input band
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            use_all_bands: If True, will use all bands rather
                than just the bands in question. Defaults to False
            apply_only_within_visits: If True, will only get corrections
                from observations within the same visit. Defaults to False
            correction_coefficients: Coefficients as a dictionary
                in the form {"A": ufloat(val, err), "tau": ufloat(val, err)}.
                ufloats here are from the uncertainties package
                Will default to values from https://arxiv.org/pdf/2512.15477
            overwrite: Whether to overwrite output files.
                Defaults to False
        """

        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs

        self.use_all_bands = use_all_bands
        self.apply_only_within_visits = apply_only_within_visits
        if correction_coefficients is None:
            correction_coefficients = CORRECTION_COEFFICIENTS
        self.correction_coefficients = correction_coefficients

        self.overwrite = overwrite

    def do_step(self):
        """Run persistence correction"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "persistence_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        # If we're using all bands, pull all the files in now
        if self.use_all_bands:
            in_dir = self.in_dir.split(os.path.sep)
            in_dir[-2] = "*"
            in_dir = os.path.sep.join(in_dir)
        else:
            in_dir = copy.deepcopy(self.in_dir)

        # FIXME: Ensure we only get MIRI image files
        files = glob.glob(
            os.path.join(
                in_dir,
                f"*mirimage_{self.step_ext}.fits",
            )
        )
        files.sort()

        # Make sure we don't duplicate files between backgrounds etc.
        # Start by just adding files from the band we're considering
        f_to_del = []
        all_files = []
        for f in files:
            # Make sure we don't remove the band we actually care about!
            band_dir = f.split(os.path.sep)[-3]

            if band_dir == self.band:
                all_files.append(os.path.basename(f))

        # Loop over again, this time removing any potential duplicates
        for f in files:

            # Make sure we don't remove the band we actually care about!
            band_dir = f.split(os.path.sep)[-3]
            if band_dir != self.band:
                if os.path.basename(f) in all_files:
                    f_to_del.append(f)
                else:
                    all_files.append(os.path.basename(f))

        for f in f_to_del:
            files.remove(f)

        success = self.persist_correct(files)

        if not success:
            log.warning("Failures detected with persistence correction")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def persist_correct(
        self,
        files,
    ):
        """Apply a persistence correction for each dither group

        Loops over dithers, applying a persistence correction
        to each image in the group

        Args:
            files: List of input files to loop over
        """

        # Get a dictionary that includes the name of the file
        # and the time of observations

        obs_dict = {}
        for f in files:
            with datamodels.open(f) as im:
                obs_t = im.meta.time.heliocentric_expmid
                band = im.meta.instrument.filter
                obs_dict[f] = {
                    "band": band,
                    "t": Time(obs_t * u.day, format="mjd"),
                }

        # If we're using all bands, now remove the other bands
        f_to_del = []
        for f in files:
            b = f.split(os.path.sep)[-3]

            if b != self.band:
                f_to_del.append(f)

        for f in f_to_del:
            files.remove(f)

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files)])

        successes = []
        with mp.get_context("fork").Pool(procs) as pool:
            for success in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_persist_correct,
                        obs_dict=obs_dict,
                    ),
                    files,
                ),
                total=len(files),
                desc="Persistence correcting files",
                ascii=True,
            ):
                successes.append(success)

        if not all(successes):
            return False

        return True

    def parallel_persist_correct(
        self,
        f,
        obs_dict=None,
    ):
        """Apply persistence correction model to data

        Will work sequentially through files within a dither group,
        taking the correction and applying an exponentially declining
        model with pre-calculated correction factors

        Args:
            f: Particular file
            obs_dict: Dictionary of form {f: obs_time}
        """

        if obs_dict is None:
            raise ValueError("Need to specify obs_dict")

        f_base = os.path.basename(f)
        # Get observation number (this is everything up to first _)
        f_obs = f_base.split("_")[0]

        # Get the time for the observation itself
        t_f = obs_dict[f]["t"]

        # Get time deltas between each file
        corr_dict = {
            f_other: {"band": t["band"], "delta_t": t_f - t["t"]}
            for f_other, t in obs_dict.items()
        }

        for f_other, vals in corr_dict.items():

            # If we only care about visits, then filter out here
            if self.apply_only_within_visits:
                f_other_base = os.path.basename(f_other)
                f_other_obs = f_other_base.split("_")[0]

                if f_obs != f_other_obs:
                    continue

            delta_t = vals["delta_t"]

            # Don't use observations from the future, or the observation itself
            if delta_t <= 0 * u.day:
                continue

            # Get the time difference in hours
            delta_t_hr = delta_t.to_value("hr")

            # Convert through to a correction factor (in %)
            corr_factor = self.correction_coefficients["A"] * unp.exp(
                self.correction_coefficients["tau"] * delta_t_hr
            )
            corr_dict[f_other].update(
                {
                    "corr_factor_val": corr_factor.nominal_value,
                    "corr_factor_err": corr_factor.std_dev,
                }
            )

        # Produce the persistence correction map
        out_f = os.path.join(self.out_dir, f_base)

        pers_corr = apply_persistence_correction(
            in_file=f,
            out_file=out_f,
            corr_dict=corr_dict,
        )

        # Produce a diagnostic plot
        self.persistence_plot(
            f=f,
            pers_corr=pers_corr,
            corr_dict=corr_dict,
        )

        return True

    def persistence_plot(
        self,
        f,
        pers_corr,
        corr_dict,
    ):
        """Create a diagnostic plot for the persistence correction

        Args:
            f: Input file
            pers_corr: Array of persistence corrections
            corr_dict: Dictionary of correction factors (in percent)
        """

        f_base = os.path.basename(f)

        # Get data and units out
        with datamodels.open(f) as im:
            data = copy.deepcopy(im.data)
            unit = im.meta.bunit_data

        # Break up by bands
        all_bands = np.unique([corr_dict[fi]["band"] for fi in corr_dict])

        # Get out useful values
        corr_factors = np.array(
            [
                [
                    corr_dict[fi].get("corr_factor_val", np.nan)
                    for fi in corr_dict
                    if corr_dict[fi]["band"] == b
                ]
                for b in all_bands
            ]
        )
        corr_factor_errs = np.array(
            [
                [
                    corr_dict[fi].get("corr_factor_err", np.nan)
                    for fi in corr_dict
                    if corr_dict[fi]["band"] == b
                ]
                for b in all_bands
            ]
        )
        delta_t_vals = np.array(
            [
                [
                    corr_dict[fi].get("delta_t", np.nan)
                    for fi in corr_dict
                    if corr_dict[fi]["band"] == b
                ]
                for b in all_bands
            ]
        )

        bands = np.array([[b] * corr_factors.shape[1] for b in all_bands])

        # Remove anything we don't have
        non_nan_idx = ~np.isnan(corr_factors)
        corr_factors = corr_factors[non_nan_idx]
        corr_factor_errs = corr_factor_errs[non_nan_idx]
        delta_t_vals = delta_t_vals[non_nan_idx]
        bands = bands[non_nan_idx]
        bands_filtered = np.unique(bands)

        delta_t_vals = np.array([t.to_value("hr") for t in delta_t_vals])

        if len(corr_factors) == 0:
            log.info(f"No correction factors found for {f}")
        else:

            plot_dir = os.path.join(self.out_dir, "plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plot_base = str(f_base.replace(".fits", ""))

            plot_name = os.path.join(plot_dir, plot_base)

            plt.figure(figsize=(12, 6))
            ax = plt.subplot(131)

            # Plot correction factors per-band
            colours = iter(plt.cm.rainbow(np.linspace(0, 1, len(all_bands))))
            for b in all_bands:

                c = next(colours)

                if b not in bands_filtered:
                    continue

                plt.errorbar(
                    delta_t_vals[bands == b],
                    corr_factors[bands == b],
                    yerr=corr_factor_errs[bands == b],
                    c=c,
                    marker="o",
                    ls="none",
                    label=b,
                )

            # Get out limits for this curve
            xlim = list(plt.xlim())
            xlim[1] *= 1.1
            xlim = [0, xlim[1]]
            plt.xlim(xlim)

            # Plot on the model line
            persistence_model_x = np.linspace(0, xlim[1], 100)
            persistence_model_y = self.correction_coefficients[
                "A"
            ].nominal_value * np.exp(
                self.correction_coefficients["tau"].nominal_value * persistence_model_x
            )

            plt.plot(
                persistence_model_x,
                persistence_model_y,
                c="r",
            )

            plt.xlabel(r"$\Delta T$ (hr)")
            plt.ylabel("Persistence correction (%)")

            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            plt.grid()

            plt.legend(loc="lower center",
                       framealpha=1,
                       fancybox=False,
                       edgecolor="k",
                       bbox_to_anchor=(0.5, 1),
                       )

            # Plot on uncorrected data
            vmin, vmax = np.nanpercentile(data, [5, 95])

            ax = plt.subplot(132)
            im = plt.imshow(
                data,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            plt.axis("off")

            plt.title("Data")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0)
            plt.colorbar(im, cax=cax, label=unit, orientation="horizontal")

            # Finally, the persistence correction
            vmin, vmax = np.nanpercentile(pers_corr, [5, 95])

            ax = plt.subplot(133)
            im = plt.imshow(
                pers_corr,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            plt.axis("off")

            plt.title("Persistence correction")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0)
            plt.colorbar(im, cax=cax, label=unit, orientation="horizontal")

            plt.tight_layout()

            plt.savefig(f"{plot_name}.png", bbox_inches="tight")
            plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")

            plt.close()

        return True
