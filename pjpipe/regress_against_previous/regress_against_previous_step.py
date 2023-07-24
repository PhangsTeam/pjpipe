import copy
import glob
import logging
import os
import warnings

import cmocean
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pypdf import PdfWriter
from reproject import reproject_interp

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def get_diff_image(
    filename,
    v_curr,
    v_prev,
    percentiles=None,
):
    """Reproject images to get a difference image

    Args:
        filename: Name of file
        v_curr: Current version
        v_prev: Previous version
        percentiles: Percentiles for diff image. Defaults to None,
            which will be [1, 99]th percentiles
    """
    if percentiles is None:
        percentiles = [1, 99]

    with fits.open(filename) as hdu1:
        prev_filename = filename.replace(v_curr, v_prev)
        if not os.path.exists(prev_filename):
            return None, None

        hdu1["SCI"].data[hdu1["SCI"].data == 0] = np.nan

        with fits.open(prev_filename) as hdu2:
            # Reproject both HDUs, just to be sure
            hdu2["SCI"].data[hdu2["SCI"].data == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data1 = reproject_interp(
                    hdu1["SCI"],
                    hdu1["SCI"].header,
                    return_footprint=False,
                )
                data2 = reproject_interp(
                    hdu2["SCI"],
                    hdu1["SCI"].header,
                    return_footprint=False,
                )

            diff = data1 - data2

            v = np.nanmax(np.abs(np.nanpercentile(diff, percentiles)))

    return diff, v


class RegressAgainstPreviousStep:
    def __init__(
        self,
        target,
        in_dir,
        curr_version,
        prev_version=None,
        overwrite=False,
    ):
        """Create diagnostic plots to regress against previous versions

        Args:
            target: Target to consider
            in_dir: Input directory
            curr_version: Current version to compare to...
            prev_version: Previous verion
            overwrite: Whether to overwrite or not. Defaults to
                False
        """

        if prev_version is None:
            raise ValueError("prev_version should be defined")

        self.target = target
        self.in_dir = in_dir
        self.curr_version = curr_version
        self.prev_version = prev_version
        self.overwrite = overwrite

        self.out_dir = os.path.join(
            self.in_dir,
            f"{self.curr_version}_to_{self.prev_version}",
        )

    def do_step(self):
        """Run previous version regression"""

        step_complete_file = os.path.join(
            self.out_dir,
            f"{self.target}_regress_against_previous_step_complete.txt",
        )

        if self.overwrite and os.path.exists(step_complete_file):
            os.remove(step_complete_file)

        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        all_files = glob.glob(
            os.path.join(self.in_dir, self.target, "*_align.fits"),
        )

        file_dict = {}

        for key in ["nircam", "miri"]:
            files = [file for file in all_files if key in os.path.split(file)[-1]]
            files.sort()
            file_dict[key] = files

        for key in file_dict:
            success = self.regress_plot(
                file_dict=file_dict,
                key=key,
            )
            # If not everything has succeeded, then return a warning
            if not success:
                log.warning("Failures detected in previous version regression")
                return False

        # Merge these all into a single pdf doc
        merged_filename = os.path.join(
            self.out_dir,
            f"{self.curr_version}_to_{self.prev_version}_comparisons_merged.pdf",
        )

        pdfs = glob.glob(
            os.path.join(
                self.out_dir,
                "*_comparison.pdf",
            )
        )
        pdfs.sort()

        with PdfWriter() as merger:
            for pdf in pdfs:
                merger.append(pdf)

            merger.write(merged_filename)
            merger.close()

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def regress_plot(self, file_dict, key):
        """Plot per-instrument comparison

        Args:
            file_dict: Dictionary of files, separated
                by instrument
            key: Instrument key
        """

        fancy_name = {
            "miri": "MIRI",
            "nircam": "NIRCam",
        }[key]

        log.info(f"Plotting up {key}")

        files = copy.deepcopy(file_dict[key])
        if len(files) > 0:
            plot_name = os.path.join(self.out_dir, f"{self.target}_{key}_comparison")

            plt.subplots(nrows=1, ncols=len(files), figsize=(4 * len(files), 4))

            for i, file in enumerate(files):
                band = os.path.split(file)[-1].split("_")[3]

                diff, v = get_diff_image(
                    file,
                    v_curr=self.curr_version,
                    v_prev=self.prev_version,
                )

                ax = plt.subplot(1, len(files), i + 1)

                if diff is None:
                    plt.text(
                        0.5,
                        0.5,
                        f"Not present in {self.prev_version}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        bbox=dict(fc="white", ec="black", alpha=0.9),
                        transform=ax.transAxes,
                    )

                    plt.axis("off")

                else:
                    vmin, vmax = -v, v

                    im = ax.imshow(
                        diff,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmocean.cm.balance,
                        origin="lower",
                        interpolation="nearest",
                    )

                    plt.axis("off")
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0)
                    plt.colorbar(im, cax=cax, label="MJy/sr")

                plt.text(
                    0.05,
                    0.95,
                    band.upper(),
                    ha="left",
                    va="top",
                    fontweight="bold",
                    bbox=dict(fc="white", ec="black", alpha=0.9),
                    transform=ax.transAxes,
                )

            plt.suptitle(f"{self.target.upper()}, {fancy_name}")

            plt.tight_layout()

            plt.savefig(f"{plot_name}.png", bbox_inches="tight")
            plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")
            plt.close()

        return True
