import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_path)

from _pya0 import *

from .mindex_info import MINDEX_INFO
from .index_manager import download_prebuilt_index, mount_image_index
from .preprocess import preprocess, preprocess_query, preprocess_text
from .preprocess import preprocess_for_transformer, use_stemmer


def from_prebuilt_index(prebuilt_index_name, verbose=True):
    try:
        index_dir = download_prebuilt_index(prebuilt_index_name, verbose=verbose)

        # mount index if it is a loop-device image
        target_index = MINDEX_INFO[prebuilt_index_name]
        if 'image_filesystem' in target_index:
            filesystem = target_index['image_filesystem']
            index_dir = mount_image_index(index_dir, filesystem)

    except ValueError as e:
        print(str(e), file=sys.stderr)
        return None

    return index_dir
