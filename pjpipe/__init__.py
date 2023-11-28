import os
import sys

# Set the CRDS server URL before any imports
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

if sys.version_info < (3, 9):
    raise ImportError("JWST requires Python 3.9 and above.")

from .apply_wcs_adjust import ApplyWCSAdjustStep
from .astrometric_align import AstrometricAlignStep
from .astrometric_catalog import AstrometricCatalogStep
from .download import DownloadStep
from .gaia_query import GaiaQueryStep
from .get_wcs_adjust import GetWCSAdjustStep
from .level_match import LevelMatchStep
from .lv1 import Lv1Step
from .lv2 import Lv2Step
from .lv3 import Lv3Step
from .lyot_mask import LyotMaskStep
from .lyot_separate import LyotSeparateStep
from .mosaic_individual_fields import MosaicIndividualFieldsStep
from .move_raw_obs import MoveRawObsStep
from .multi_tile_destripe import MultiTileDestripeStep
from .psf_model import PSFModelStep
from .release import ReleaseStep
from .regress_against_previous import RegressAgainstPreviousStep
from .pipeline import PJPipeline
from .single_tile_destripe import SingleTileDestripeStep
from .utils import load_toml
from .anchoring import AnchoringStep
from .psf_matching import PSFMatchingStep

__all__ = [
    "AnchoringStep",
    "ApplyWCSAdjustStep",
    "AstrometricAlignStep",
    "AstrometricCatalogStep",
    "DownloadStep",
    "GaiaQueryStep",
    "GetWCSAdjustStep",
    "LevelMatchStep",
    "Lv1Step",
    "Lv2Step",
    "Lv3Step",
    "LyotMaskStep",
    "LyotSeparateStep",
    "MosaicIndividualFieldsStep",
    "MoveRawObsStep",
    "MultiTileDestripeStep",
    "PJPipeline",
    "PSFMatchingStep",
    "PSFModelStep",
    "ReleaseStep",
    "RegressAgainstPreviousStep",
    "SingleTileDestripeStep",
    "load_toml",
]


def list_steps():
    for cls in __all__:
        if "Step" in cls:
            print(cls)
