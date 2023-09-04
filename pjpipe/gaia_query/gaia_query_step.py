import logging
import os

import astropy.units as u
from astropy.coordinates import name_resolve
from astroquery.gaia import Gaia

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class GaiaQueryStep:
    def __init__(
        self,
        target,
        out_dir,
        row_limit=-1,
        radius=15,
        overwrite=False,
    ):
        """Query Gaia to get sources for astrometric alignment

        Args:
            target: Target to get GAIA sources for
            out_dir: Where to save GAIA catalog to
            row_limit: Maximum number of rows to return.
                Defaults to -1, which will return everything
            radius: Radius in arcmin to query. Defaults to
                15
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        radius = radius * u.arcmin

        self.target = target
        self.out_dir = out_dir
        self.row_limit = row_limit
        self.radius = radius
        self.overwrite = overwrite

    def do_step(self):
        """Run Gaia catalog query"""

        out_name = os.path.join(
            self.out_dir,
            f"Gaia_DR3_{self.target}.fits",
        )

        if self.overwrite:
            if os.path.exists(out_name):
                os.remove(out_name)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            f"{self.target}_gaia_query_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        success = self.make_catalog(out_name=out_name)

        if not success:
            log.warning("Failures detected in GAIA query")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def make_catalog(
        self,
        out_name,
    ):
        """Create a Gaia catalog

        Args:
            out_name: Name to save catalog to
        """

        log.info(f"Resolving target {self.target}")
        try:
            coords = name_resolve.get_icrs_coordinates(self.target)
        except name_resolve.NameResolveError:
            log.warning(f"Unable to resolve {self.target}")
            return False

        jobstr = f"SELECT"
        if self.row_limit != -1:
            jobstr += f" TOP {self.row_limit}"
        jobstr += " * FROM gaiadr3.gaia_source\n"
        jobstr += "WHERE 1=CONTAINS(POINT('ICRS', gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),"
        jobstr += f"CIRCLE('ICRS',{coords.ra.deg},{coords.dec.deg},{self.radius.to(u.deg).value}))\n"
        jobstr += "ORDER by gaiadr3.gaia_source.phot_bp_mean_mag ASC"

        log.info("Launching job query to Gaia archive with query:")
        log.info(jobstr)

        job = Gaia.launch_job_async(jobstr, dump_to_file=False)
        results = job.get_results()
        removelist = []

        # Strip object columns from FITS table
        for col in results.columns:
            if results[col].dtype == "object":
                removelist += [col]
        results.remove_columns(removelist)
        results.write(out_name, overwrite=True)

        return True
