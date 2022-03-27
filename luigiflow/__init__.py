__version__ = "0.1.0"

import os

# This parameter needs to be set before loading any luigi module.
os.environ["LUIGI_CONFIG_PARSER"] = "toml"
