import copy
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from functools import partial

import jwst
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from jwst.pipeline import calwebb_image2
from stdatamodels.jwst import datamodels

from ..utils import get_band_type, get_obs_table, attribute_setter, save_file

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

BGR_CHECK_TYPES = [
    "parallel_off",
    "check_in_name",
    "filename",
]


class Lv2Step:
    def __init__(
        self,
        target,
        band,
        in_dir,
        out_dir,
        dr_version,
        step_ext,
        is_bgr,
        procs,
        bgr_check_type="parallel_off",
        bgr_background_name="off",
        bgr_observation_types=None,
        process_bgr_like_science=False,
        jwst_parameters=None,
        updated_flats_dir=None,
        overwrite=False,
    ):
        """Wrapper around the level 2 JWST pipeline

        Args:
            target: Target to consider
            band: Band to consider
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            is_bgr: Whether we're processing background observations or not
            procs: Number of processes to run in parallel
            bgr_check_type: Method to check if obs is science
                or background. Options are given by BGR_CHECK_TYPES.
                Defaults to 'parallel_off'
            bgr_background_name: If `bgr_check_type` is 'check_in_name'
                or 'filename', this is the string to match
            bgr_observation_types: List of observation types with dedicated
                backgrounds. Defaults to None, i.e. no observations have
                backgrounds
            process_bgr_like_science: If True, will process background
                images as if they are science images. Defaults to False
            jwst_parameters: Parameter dictionary to pass to
                the level 2 pipeline. Defaults to None,
                which will run the observatory defaults
            updated_flats_dir: Directory with the updated flats to use
                instead of default ones. Defaults to None, which will
                use the pipeline default flats
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if bgr_observation_types is None:
            bgr_observation_types = []
        if jwst_parameters is None:
            jwst_parameters = {}

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.dr_version = dr_version
        self.step_ext = step_ext
        self.is_bgr = is_bgr
        self.procs = procs
        self.bgr_check_type = bgr_check_type
        self.bgr_background_name = bgr_background_name
        self.bgr_observation_types = bgr_observation_types
        self.process_bgr_like_science = process_bgr_like_science
        self.jwst_parameters = jwst_parameters
        self.updated_flats_dir = updated_flats_dir
        self.overwrite = overwrite

        self.band_type = get_band_type(self.band)

    def do_step(self):
        """Run the level 2 pipeline"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)
            os.system(f"rm -rf {os.path.join(self.in_dir, '*.json')}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "lv2_step_complete.txt",
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

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files)])

        asn_files = self.create_asn_files(
            files=files,
            procs=procs,
        )

        successes = self.run_step(
            asn_files,
            procs=procs,
        )

        # If not everything has succeeded, then return a warning
        if not np.all(successes):
            log.warning("Failures detected in level 2 pipeline")
            return False

        # Make sure to propagate PJPipe info into the output files
        self.propagate_metadata(files)

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def create_asn_files(
        self,
        files,
        procs=1,
    ):
        """Setup asn lv2 files

        Args:
            * files: List of files to include in the asn file
            * procs: If greater than 1, will write out an asn file
                per observation. Defaults to 1
        """

        log.info("Building asn files")

        check_bgr = True

        # If we have NIRCam operating with parallel offs, switch off background checking
        # else everything will be flagged as backgrounds
        if self.band_type == "nircam" and self.bgr_check_type == "parallel_off":
            check_bgr = False

        tab = get_obs_table(
            files=files,
            check_bgr=check_bgr,
            check_type=self.bgr_check_type,
            background_name=self.bgr_background_name,
        )
        tab.sort(keys="Start")

        # Loop over science first, then backgrounds
        sci_tab = tab[tab["Type"] == "sci"]
        bgr_tab = tab[tab["Type"] == "bgr"]

        asn_files = []

        # If we're processing backgrounds like science,
        # join the tables
        if self.process_bgr_like_science and not self.is_bgr:
            full_tab = vstack([sci_tab, bgr_tab])
            full_tab["Type"] = "sci"

        # If we're processing background observations,
        # then just take the background table and don't
        # include those backgrounds again
        elif self.is_bgr:
            full_tab = copy.deepcopy(bgr_tab)
            full_tab["Type"] = "sci"
            bgr_tab = []

        # Or, just take the science
        else:
            full_tab = copy.deepcopy(sci_tab)

        if procs > 1:
            for row_id, row in enumerate(full_tab):
                asn_file = os.path.join(
                    self.in_dir,
                    f"asn_lv2_{self.band}_{row_id}.json",
                )

                # Make the row a table so we can iterate
                row = Table(row)
                json_content = self.get_asn_json(
                    sci_tab=row,
                    bgr_tab=bgr_tab,
                )

                with open(asn_file, "w+") as f:
                    json.dump(json_content, f)

                asn_files.append(asn_file)

        else:
            asn_file = os.path.join(
                self.in_dir,
                f"asn_lv2_{self.band}.json",
            )
            json_content = self.get_asn_json(
                sci_tab=full_tab,
                bgr_tab=bgr_tab,
            )

            with open(asn_file, "w+") as f:
                json.dump(json_content, f)

            asn_files.append(asn_file)

        return asn_files

    def get_asn_json(
        self,
        sci_tab,
        bgr_tab,
    ):
        """Build the JSON file from the science and background tables"""

        json_content = {
            "asn_type": "image2",
            "asn_rule": "DMSLevel2bBase",
            "version_id": time.strftime("%Y%m%dt%H%M%S"),
            "code_version": jwst.__version__,
            "degraded_status": "No known degraded exposures in association.",
            "program": sci_tab["Program"][0],
            "constraints": "none",
            "asn_id": f"o{sci_tab['Obs_ID'][0]}",
            "asn_pool": "none",
            "products": [],
        }

        # Pull out an average background time if we care about that
        if self.band_type in self.bgr_observation_types:
            bgr_times = []
            for row in bgr_tab:
                with datamodels.open(row["File"]) as im:
                    bgr_times.append(im.meta.time.heliocentric_expmid)
            if len(bgr_times) == 0:
                bgr_time = "null"
            else:
                bgr_time = np.nanmean(bgr_times)
        else:
            bgr_time = "null"

        for row in sci_tab:

            # Get the difference from the backgrounds, if we have any
            if bgr_time != "null":
                with datamodels.open(row["File"]) as im:
                    sci_bgr_offset = im.meta.time.heliocentric_expmid - bgr_time
            else:
                sci_bgr_offset = "null"

            name = os.path.split(row["File"])[-1].split("_rate.fits")[0]

            json_content["products"].append(
                {
                    "name": name,
                    "array": row["Array"],
                    "members": [
                        {
                            "expname": row["File"],
                            "exptype": "science",
                            "exposerr": "null",
                            "scibgroffset": str(sci_bgr_offset),
                        }
                    ],
                }
            )

        if self.band_type in self.bgr_observation_types:
            for product in json_content["products"]:
                for row in bgr_tab:
                    product["members"].append(
                        {
                            "expname": row["File"],
                            "exptype": "background",
                            "exposerr": "null",
                        }
                    )

        return json_content

    def run_step(
        self,
        asn_files,
        procs=1,
    ):
        """Wrap parallelism around the level 2 pipeline

        Args:
            asn_files: List of association files to loop over
            procs: Number of processes to run. Defaults to 1
        """

        log.info("Running level 2 pipeline")

        # Ensure we pre-cache references, to avoid errors in multiprocessing. Loop over
        # all just to be safe
        for asn_file in asn_files:
            config = calwebb_image2.Image2Pipeline.get_config_from_reference(asn_file)
            im2 = calwebb_image2.Image2Pipeline.from_config_section(config)
            im2._precache_references(asn_file)

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in pool.imap_unordered(
                partial(
                    self.parallel_lv2,
                ),
                asn_files,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_lv2(
        self,
        asn_file,
    ):
        """Parallelise running lv2 processing

        Args:
            asn_file: Association files to run
        """

        config = calwebb_image2.Image2Pipeline.get_config_from_reference(asn_file)
        im2 = calwebb_image2.Image2Pipeline.from_config_section(config)
        im2.output_dir = self.out_dir

        im2 = attribute_setter(
            im2,
            parameters=self.jwst_parameters,
            band=self.band,
            target=self.target,
        )

        if self.updated_flats_dir is not None:
            my_flat = [
                f
                for f in glob.glob(os.path.join(self.updated_flats_dir, "*.fits"))
                if self.band in f
            ]
            if len(my_flat) != 0:
                im2.flat_field.user_supplied_flat = my_flat[0]

        # Run the level 2 pipeline
        im2.run(asn_file)

        del im2
        gc.collect()

        # Add in background offset times, if we're a background observation
        if self.band_type in self.bgr_observation_types:
            with open(asn_file) as f:
                asn_json = json.load(f)
                for product in asn_json["products"]:
                    for member in product["members"]:

                        # If we're not a science image, or we don't have a background, skip
                        if member["exptype"] != "science":
                            continue
                        if member["scibgroffset"] == "null":
                            continue

                        expname_short = member["expname"].split(os.path.sep)[-1].split("_rate.fits")[0]
                        out_files = glob.glob(os.path.join(self.out_dir,
                                                           f"{expname_short}*cal.fits",
                                                           ),
                                              )
                        sci_bgr_offset = member["scibgroffset"]
                        for out_file in out_files:
                            with datamodels.open(out_file) as im:
                                im.meta.time.sci_bgr_offset = sci_bgr_offset
                                im.save(out_file)
                            with fits.open(out_file) as hdu:
                                hdu[0].header["DT_BGR"] = (sci_bgr_offset, 'Time offset between image and backgrounds')
                                hdu.writeto(out_file, overwrite=True)

        return True

    def propagate_metadata(self,
                           files,
                           ):
        """Propagate metadata through to the output files

        Args:
            files: List of files to loop over
        """

        for file in files:

            file_split = file.split(os.path.sep)[-1].replace(f"{self.step_ext}.fits",
                                                             "cal.fits",
                                                             )
            out_name = os.path.join(self.out_dir,
                                    file_split,
                                    )

            # If the file doesn't exist (e.g. it's a background file that won't come through),
            # then skip
            if not os.path.exists(out_name):
                continue

            with datamodels.open(out_name) as im:
                save_file(im, out_name=out_name, dr_version=self.dr_version)

            del im

        return True
