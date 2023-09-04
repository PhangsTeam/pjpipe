import os

import pjpipe

this_dir = os.path.dirname(os.path.realpath(__file__))

config_file = os.path.join(this_dir, "config.toml")
local_file = os.path.join(this_dir, "cedar.toml")

# We need to set CRDS path
local = pjpipe.load_toml(local_file)
crds_path = local["crds_path"]
os.environ["CRDS_PATH"] = crds_path
if not os.path.exists(crds_path):
    os.makedirs(crds_path)

# If this is our first time running things, we need to pull some
# references to avoid fatal errors
crds_files = os.listdir(crds_path)
if "config" not in crds_files:
    os.system("crds sync --jwst")

pjp = pjpipe.PJPipeline(
    config_file=config_file,
    local_file=local_file,
)
pjp.do_pipeline()
