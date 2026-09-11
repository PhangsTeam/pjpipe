import logging
import os
import sys
from datetime import datetime
from importlib.metadata import version
from logging.handlers import RotatingFileHandler

# Set the CRDS server URL before any imports
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

if sys.version_info < (3, 11):
    raise ImportError("JWST requires Python 3.11 and above.")

# Get the version
__version__ = version("pjpipe")

# Set up the logger
log = logging.getLogger(__name__)
dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"pjpipe_{dt}.log"
log_level = "INFO"
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(),
                    ]
                    )

from .anchoring import AnchoringStep
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
from .persistence import PersistenceStep
from .pipeline import PJPipeline
from .psf_matching import PSFMatchingStep
from .single_tile_destripe import SingleTileDestripeStep
from .utils import load_toml

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
    "PersistenceStep",
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
