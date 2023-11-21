import logging
import os
import shutil
import glob

from astropy.io import fits
from tqdm import tqdm

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class ReleaseStep:
    def __init__(
        self,
        in_dir,
        out_dir,
        target,
        bands,
        file_exts=None,
        remove_bloat=False,
        move_tweakback=False,
        move_backgrounds=False,
        move_individual_fields=False,
        move_psf_matched=False,
        move_diagnostic_plots=False,
        compress_diagnostic_plots=True,
        lv3_dir="lv3",
        tweakback_dir="lv3",
        tweakback_ext="tweakback",
        background_dir="lv2",
        background_ext="combinedbackground",
        individual_fields_dir="mosaic_individual_fields",
        psf_matched_dir="psf_matching",
        diagnostic_plot_dir="plots",
        overwrite=False,
    ):
        """Tidies up files, moves to a single directory for release

        This step will move the final useful files, plus optionally
        any tweakback'd crf files and background files, into a neat
        directory structure for release

        Args:
            in_dir: Input directory
            out_dir: Output directory
            target: Target to consider
            bands: Bands to consider
            file_exts: List of filetypes to move. Defaults to moving fits
                files, plus any generated catalogues and segmentation maps
            remove_bloat: Will remove generally un-needed extensions from
                fits files. Defaults to False
            move_tweakback: Whether to move tweakback'd crf files or not.
                Defaults to False
            move_backgrounds: Whether to move combined background files or not.
                Defaults to False
            move_individual_fields: Whether to move individual field mosaics
                or not. Defaults to False
            move_psf_matched: Whether to move PSF matched images or not.
                Defaults to False
            move_diagnostic_plots: Whether to move various diagnostic plots or not.
                Defaults to False
            compress_diagnostic_plots: Whether to compress the diagnostic plot folder
                to limit file number. Defaults to True
            lv3_dir: Where level 3 files are located, relative
                to the target directory structure. Defaults to "lv3"
            background_dir: Where tweakback files are located, relative
                to the target directory structure. Defaults to "lv3"
            tweakback_ext: Filename extension for tweakback files. Defaults to
                "tweakback"
            background_dir: Where combined background files are located, relative
                to the target directory structure. Defaults to "lv2"
            background_ext: Filename extension for combined background files.
                Defaults to "combinedbackground"
            individual_fields_dir: Where individual field mosaics are located,
                relative to the target directory structure. Defaults to
                "mosaic_individual_fields"
            overwrite: Whether to overwrite or not. Defaults to False
        """

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.target = target
        self.bands = bands
        self.remove_bloat = remove_bloat
        self.move_tweakback = move_tweakback
        self.move_backgrounds = move_backgrounds
        self.move_individual_fields = move_individual_fields
        self.move_psf_matched = move_psf_matched
        self.move_diagnostic_plots = move_diagnostic_plots
        self.compress_diagnostic_plots = compress_diagnostic_plots
        self.overwrite = overwrite

        self.hdu_ext_to_delete = [
            # 'ERR',
            # 'CON',
            # 'WHT',
            "VAR_POISSON",
            "VAR_RNOISE",
            "VAR_FLAT",
        ]

        if file_exts is None:
            file_exts = [
                "i2d.fits",
                "i2d_align.fits",
                "i2d_anchor.fits",
                "i2d_align_table.fits",
                "cat.ecsv",
                "astro_cat.fits",
                "segm.fits",
            ]
        self.file_exts = file_exts
        self.lv3_dir = lv3_dir
        self.tweakback_dir = tweakback_dir
        self.tweakback_ext = tweakback_ext
        self.background_dir = background_dir
        self.background_ext = background_ext
        self.individual_fields_dir = individual_fields_dir
        self.psf_matched_dir = psf_matched_dir
        self.diagnostic_plot_dir = diagnostic_plot_dir

    def do_step(self):
        """Run the release step"""

        step_complete_file = os.path.join(
            self.in_dir,
            "release_step_complete.txt",
        )

        out_dir = os.path.join(self.out_dir, self.target)

        if self.overwrite:
            if os.path.exists(step_complete_file):
                os.remove(step_complete_file)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Check if we've already run the step
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        # Move the anchor tables, if they exist
        anchor_tabs = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_anchor_tab.fits",
            )
        )
        for anchor_tab in anchor_tabs:
            out_name = os.path.join(
                out_dir,
                os.path.split(anchor_tab)[-1],
            )
            os.system(f"cp {anchor_tab} {out_name}")

        for band in tqdm(
            self.bands,
            ascii=True,
            desc="Looping over bands",
        ):
            for file_ext in tqdm(
                self.file_exts,
                ascii=True,
                desc="Looping over file types",
                leave=False,
            ):
                self.move_files(
                    band=band,
                    file_ext=file_ext,
                )
            if self.move_tweakback:
                self.do_move_tweakback(
                    band=band,
                )
            if self.move_backgrounds:
                self.do_move_backgrounds(
                    band=band,
                )
            if self.move_individual_fields:
                self.do_move_individual_fields(
                    band=band,
                )
            if self.move_psf_matched:
                self.do_move_psf_matched(
                    band=band,
                )
            if self.move_diagnostic_plots:
                self.do_move_diagnostic_plots(
                    band=band,
                )

        if self.move_diagnostic_plots and self.compress_diagnostic_plots:
            self.do_compress_diagnostic_plots()

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def move_files(
        self,
        band,
        file_ext,
    ):
        """Move files

        Args:
            band: Band to consider
            file_ext: File extension to move
        """

        files = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                self.lv3_dir,
                f"*_{file_ext}",
            )
        )

        if len(files) == 0:
            return True

        files.sort()

        out_dir = os.path.join(self.out_dir, self.target)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in files:
            out_name = os.path.join(
                out_dir,
                os.path.split(file)[-1],
            )

            if (
                file_ext in ["i2d.fits", "i2d_align.fits", "i2d_anchor.fits"]
                and self.remove_bloat
            ):
                # For these, we want to pull out only the data and error extensions. Everything else
                # is just bloat
                with fits.open(file, memmap=False) as hdu:
                    for hdu_ext in self.hdu_ext_to_delete:
                        del hdu[hdu_ext]

                    hdu.writeto(out_name, overwrite=True)
                    del hdu

            else:
                os.system(f"cp {file} {out_name}")

        return True

    def do_move_tweakback(
        self,
        band,
    ):
        """Move tweakback crf files

        Args:
            band: Band to consider
        """

        files = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                self.tweakback_dir,
                f"*_{self.tweakback_ext}.fits",
            )
        )

        if len(files) == 0:
            return True

        files.sort()

        out_dir = os.path.join(
            self.out_dir,
            self.target,
            f"{band.lower()}_tweakback",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(
            files,
            ascii=True,
            desc="tweakback",
            leave=False,
        ):
            os.system(f"cp {file} {out_dir}")

        return True

    def do_move_backgrounds(
        self,
        band,
    ):
        """Move combined background files

        Args:
            band: Band to consider
        """

        files = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                self.background_dir,
                f"*_{self.background_ext}.fits",
            )
        )

        if len(files) == 0:
            return True

        files.sort()

        out_dir = os.path.join(
            self.out_dir,
            self.target,
            f"{band.lower()}_background",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(
            files,
            ascii=True,
            desc="background",
            leave=False,
        ):
            os.system(f"cp {file} {out_dir}")

        return True

    def do_move_individual_fields(
        self,
        band,
    ):
        """Move individual field mosaics

        Args:
            band: Band to consider
        """

        files = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                self.individual_fields_dir,
                f"*.fits",
            )
        )

        if len(files) == 0:
            return True

        files.sort()

        out_dir = os.path.join(
            self.out_dir,
            self.target,
            f"{band.lower()}_indiv_fields",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(
            files,
            ascii=True,
            desc="Individual fields",
            leave=False,
        ):
            os.system(f"cp {file} {out_dir}")

        return True

    def do_move_psf_matched(
        self,
        band,
    ):
        """Move PSF matched images

        Args:
            band: Band to consider
        """

        files = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                self.psf_matched_dir,
                f"*.fits",
            )
        )

        if len(files) == 0:
            return True

        files.sort()

        out_dir = os.path.join(
            self.out_dir,
            self.target,
            f"{band.lower()}_psf_matched",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(
            files,
            ascii=True,
            desc="PSF Matched",
            leave=False,
        ):
            out_name = os.path.join(
                out_dir,
                os.path.split(file)[-1],
            )

            if self.remove_bloat:
                # For these, we want to pull out only the data and error extensions. Everything else
                # is just bloat
                with fits.open(file, memmap=False) as hdu:
                    for hdu_ext in self.hdu_ext_to_delete:
                        del hdu[hdu_ext]

                    hdu.writeto(out_name, overwrite=True)
                    del hdu
            else:
                os.system(f"cp {file} {out_name}")

        return True

    def do_move_diagnostic_plots(
        self,
        band,
    ):
        """Move various diagnostic plots

        Args:
            band: Band to consider
        """

        # We can start with the anchoring, since the plots there are a bit different
        files_anchoring_png = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                "anchoring",
                f"*.png",
            )
        )
        files_anchoring_pdf = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                "anchoring",
                f"*.pdf",
            )
        )

        # Now for other step plots
        files_png = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                "*",
                self.diagnostic_plot_dir,
                f"*.png",
            )
        )
        files_pdf = glob.glob(
            os.path.join(
                self.in_dir,
                band,
                "*",
                self.diagnostic_plot_dir,
                f"*.pdf",
            )
        )

        files = files_anchoring_png + files_anchoring_pdf + files_png + files_pdf

        if len(files) == 0:
            return True

        files.sort()

        # Anchoring plots go elsewhere
        out_anchoring_dir = os.path.join(
            self.out_dir,
            self.target,
            f"anchoring_diagnostic_plots",
        )
        if not os.path.exists(out_anchoring_dir):
            os.makedirs(out_anchoring_dir)

        out_dir = os.path.join(
            self.out_dir,
            self.target,
            f"{band.lower()}_diagnostic_plots",
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(
                files,
                ascii=True,
                desc="Diagnostic plots",
                leave=False,
        ):
            # Split this out by step so the folders aren't crazy
            file_split = file.split(os.path.sep)
            if "anchoring" in file_split:
                full_out_dir = os.path.join(out_anchoring_dir)
            else:
                full_out_dir = os.path.join(out_dir, file_split[-3])
            if not os.path.exists(full_out_dir):
                os.makedirs(full_out_dir)

            out_name = os.path.join(
                full_out_dir,
                file_split[-1],
            )
            os.system(f"cp {file} {out_name}")

        return True

    def do_compress_diagnostic_plots(self):
        """Compress diagnostic plot directories"""

        orig_dir = os.getcwd()

        out_dir = os.path.join(self.out_dir, self.target)

        os.chdir(out_dir)

        plot_dirs = glob.glob("*_diagnostic_plots")

        for plot_dir in tqdm(
                plot_dirs,
                ascii=True,
                desc="Compressing diagnostic plots",
                leave=False,
        ):
            os.system(f"tar -czf {plot_dir}.tar.gz {plot_dir}")
            os.system(f"rm -rf {plot_dir}")

        os.chdir(orig_dir)

        return True
