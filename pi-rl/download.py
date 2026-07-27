# adapted from openpi
import os

from openpi.shared import download


os.environ["OPENPI_DATA_HOME"] = "/path/to/openpi_data_home"

download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")