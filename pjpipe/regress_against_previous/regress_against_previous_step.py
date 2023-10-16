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
    curr_file_ext,
    percentiles=None,
    file_exts=None,
):
    """Reproject images to get a difference image

    Args:
        filename: Name of file
        v_curr: Current version
        v_prev: Previous version
        curr_file_ext: Current file extension
        percentiles: Percentiles for diff image. Defaults to None,
            which will be [1, 99]th percentiles
        file_exts: List of file extensions to search for the previous
            file in priority order. Defaults to None, which will go
            anchor->align->pipeline.
    """
    if percentiles is None:
        percentiles = [1, 99]
    if file_exts is None:
        file_exts = [
            "i2d_anchor.fits",
            "i2d_align.fits",
            "i2d.fits",
        ]

    with fits.open(filename) as hdu1:
        prev_file_found = False

        for file_ext in file_exts:
            if not prev_file_found:
                prev_filename = filename.replace(v_curr, v_prev)
                prev_filename = prev_filename.replace(curr_file_ext, file_ext)
                if os.path.exists(prev_filename):
                    prev_file_found = True

        if not prev_file_found:
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
        file_exts=None,
        overwrite=False,
    ):
        """Create diagnostic plots to regress against previous versions

        Args:
            target: Target to consider
            in_dir: Input directory
            curr_version: Current version to compare to...
            prev_version: Previous version
            file_exts: File extensions (in priority order) to search for
            overwrite: Whether to overwrite or not. Defaults to
                False
        """

        if prev_version is None:
            raise ValueError("prev_version should be defined")

        if file_exts is None:
            file_exts = [
                "i2d_anchor.fits",
                "i2d_align.fits",
                "i2d.fits",
            ]

        self.target = target
        self.in_dir = in_dir
        self.curr_version = curr_version
        self.prev_version = prev_version
        self.file_exts = file_exts
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

        # Get list of appropriate best files
        all_files = []
        all_file_exts = []
        for file_ext in self.file_exts:
            # Get all the files that match
            files = glob.glob(
                os.path.join(self.in_dir, self.target, f"*_{file_ext}"),
            )

            # If they're already in the list, don't add them again
            filtered_files = []

            for file in files:
                file_already_in_list = False

                file_short = os.path.split(file)[-1]
                file_short = "_".join(file_short.split("_")[:-2])

                for list_file in all_files:
                    if not file_already_in_list:
                        if file_short in list_file:
                            file_already_in_list = True

                if not file_already_in_list:
                    filtered_files.append(file)

            # Add to the final list
            all_files.extend(filtered_files)
            all_file_exts.extend([file_ext] * len(filtered_files))

        file_dict = {}

        for key in ["nircam", "miri"]:
            idx = [
                i
                for i in range(len(all_files))
                if key in os.path.split(all_files[i])[-1]
            ]
            files = [all_files[i] for i in idx]
            file_exts = [all_file_exts[i] for i in idx]
            sort_idx = np.argsort(files)
            files = np.asarray(files)[sort_idx]
            file_exts = np.asarray(file_exts)[sort_idx]
            file_dict[key] = {
                "files": files,
                "file_exts": file_exts,
            }

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

        files = copy.deepcopy(file_dict[key]["files"])
        file_exts = copy.deepcopy(file_dict[key]["file_exts"])
        if len(files) > 0:
            plot_name = os.path.join(self.out_dir, f"{self.target}_{key}_comparison")

            plt.subplots(nrows=1, ncols=len(files), figsize=(4 * len(files), 4))

            for i, file in enumerate(files):
                file_short = os.path.split(file)[-1]

                # Make sure we get bands right if it's a background obs
                if "_bgr_" in file_short:
                    is_bgr = True
                else:
                    is_bgr = False

                band = file_short.split("_")[3]
                file_ext = file_exts[i]

                diff, v = get_diff_image(
                    file,
                    v_curr=self.curr_version,
                    v_prev=self.prev_version,
                    curr_file_ext=file_ext,
                    file_exts=self.file_exts,
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

                band_text = band.upper()
                if is_bgr:
                    band_text += " bgr"

                plt.text(
                    0.05,
                    0.95,
                    band_text,
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
