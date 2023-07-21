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
        progress_dict,
        file_exts=None,
        remove_bloat=True,
        move_tweakback=False,
        move_backgrounds=False,
        tweakback_dir="lv3",
        tweakback_ext="tweakback",
        background_dir="lv2",
        background_ext="combinedbackground",
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
            progress_dict: The progress dictionary the pipeline builds up.
                This is used to figure out what subdirectories we should
                be looking in
            file_exts: List of filetypes to move. Defaults to moving fits
                files, plus any generated catalogues and segmentation maps
            remove_bloat: Will remove generally un-needed extensions from
                fits files. Defaults to True
            move_tweakback: Whether to move tweakback'd crf files or not.
                Defaults to False
            move_backgrounds: Whether to move combined background files or not.
                Defaults to False
            background_dir: Where tweakback files are located, relative
                to the target directory structure. Defaults to "lv3"
            tweakback_ext: Filename extension for tweakback files. Defaults to
                "tweakback"
            background_dir: Where combined background files are located, relative
                to the target directory structure. Defaults to "lv2"
            background_ext: Filename extension for combined background files.
                Defaults to "combinedbackground"
            overwrite: Whether to overwrite or not. Defaults to False
        """

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.target = target
        self.bands = bands
        self.progress_dict = progress_dict
        self.remove_bloat = remove_bloat
        self.move_tweakback = move_tweakback
        self.move_backgrounds = move_backgrounds
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
                "i2d_align_table.fits",
                "cat.ecsv",
                "astro_cat.fits",
                "segm.fits",
            ]
        self.file_exts = file_exts
        self.tweakback_dir = tweakback_dir
        self.tweakback_ext = tweakback_ext
        self.background_dir = background_dir
        self.background_ext = background_ext

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

        band_dir = self.progress_dict[self.target][band]["dir"]

        files = glob.glob(
            os.path.join(
                band_dir,
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

            if file_ext in ["i2d.fits", "i2d_align.fits"] and self.remove_bloat:
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