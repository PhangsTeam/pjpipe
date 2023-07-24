import copy
import glob
import logging
import os
import shutil
import fnmatch

from stdatamodels.jwst import datamodels
from tqdm import tqdm

from ..utils import get_band_ext, get_band_type

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class MoveRawObsStep:
    def __init__(
        self,
        target,
        band,
        step_ext,
        in_dir,
        out_dir,
        obs_to_skip=None,
        extra_obs_to_include=None,
        overwrite=False,
    ):
        """Move raw observations from the MAST folder into a specific target/band folder

        Args:
            target: Target to consider
            band: Band to consider
            step_ext: .fits extension to look for (e.g. uncal, rate etc.)
            in_dir: Input directory to search for files (should be some kind of mastDownload-esque
                folder)
            out_dir: Where to move files to
            obs_to_skip: List of failed or otherwise observations that shouldn't be moved.
                Defaults to None, which skips nothing
            extra_obs_to_include: List of extra observations to include, for example MIRI
                flats from elsewhere. Defaults to None, which includes nothing extra
            overwrite (bool): Whether to overwrite or not. Defaults to False
        """

        if target is None:
            raise ValueError("target should be specified")
        if band is None:
            raise ValueError("band should be specified")
        if step_ext is None:
            raise ValueError("step_ext should be specified")

        self.target = target
        self.band = band
        self.step_ext = step_ext
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.obs_to_skip = obs_to_skip
        self.extra_obs_to_include = extra_obs_to_include
        self.overwrite = overwrite

        self.band_type = get_band_type(band)
        self.band_ext = get_band_ext(band)

    def do_step(self):
        """Move raw observation files"""

        step_complete_file = os.path.join(
            self.out_dir,
            "move_raw_obs_step_complete.txt",
        )

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        raw_files = self.get_raw_files()

        if len(raw_files) == 0:
            log.warning("No raw files found. Skipping")
            return False

        raw_files.sort()

        success = self.move_raw_files(raw_files)

        # If not everything has succeeded, then return a warning
        if not success:
            log.warning("Failures detected in level 2 pipeline")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def get_raw_files(self):
        """Build a list of raw files"""

        raw_files = glob.glob(
            os.path.join(
                self.in_dir,
                self.target,
                "mastDownload",
                "JWST",
                f"*{self.band_ext}",
                f"*{self.band_ext}_{self.step_ext}.fits",
            )
        )
        raw_files.sort()

        if self.extra_obs_to_include is not None:
            for other_target in self.extra_obs_to_include:
                for obs_to_include in self.extra_obs_to_include[other_target]:
                    extra_files = glob.glob(
                        os.path.join(
                            self.in_dir,
                            other_target,
                            "mastDownload",
                            "JWST",
                            f"{obs_to_include}*{self.band_ext}",
                            f"*{self.band_ext}_{self.step_ext}.fits",
                        )
                    )
                    extra_files.sort()

                    raw_files.extend(extra_files)

        return raw_files

    def move_raw_files(
        self,
        raw_files,
    ):
        """Actually move the raw files"""

        for raw_file in tqdm(
            raw_files,
            ascii=True,
            desc="Moving raw files",
        ):
            raw_fits_name = raw_file.split(os.path.sep)[-1]
            hdu_out_name = os.path.join(self.out_dir, raw_fits_name)

            # If we have a failed observation, skip it
            skip_file = False
            for obs in self.obs_to_skip:
                match = fnmatch.fnmatch(raw_fits_name, obs)
                if match:
                    skip_file = True
            if skip_file:
                continue

            if not os.path.exists(hdu_out_name) or self.overwrite:
                with datamodels.open(raw_file) as im:
                    # Check we've got the right filter
                    hdu_filter = im.meta.instrument.filter

                    if self.band_type == "nircam":
                        hdu_pupil = im.meta.instrument.pupil

                        pupil = "CLEAR"
                        jwst_filter = copy.deepcopy(self.band)

                        # For some NIRCAM filters, we need to distinguish filter/pupil.
                        # TODO: These may not be unique, so may need editing
                        if self.band in ["F162M", "F164N"]:
                            pupil = copy.deepcopy(self.band)
                            jwst_filter = "F150W2"
                        if self.band == "F323N":
                            pupil = copy.deepcopy(self.band)
                            jwst_filter = "F322W2"
                        if self.band in ["F405N", "F466N", "F470N"]:
                            pupil = copy.deepcopy(self.band)
                            jwst_filter = "F444W"

                        if hdu_filter == jwst_filter and hdu_pupil == pupil:
                            im.save(hdu_out_name)

                    elif self.band_type == "miri":
                        if hdu_filter == self.band:
                            im.save(hdu_out_name)

                del im

        return True
