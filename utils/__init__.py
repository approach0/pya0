import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_path)

from _pya0 import *

from .mindex_info import MINDEX_INFO
from .index_manager import download_prebuilt_index, mount_image_index
from .preprocess import preprocess, preprocess_query


def from_prebuilt_index(prebuilt_index_name):
    print(f'Attempting to initialize pre-built index {prebuilt_index_name}.')
    try:
        index_dir = download_prebuilt_index(prebuilt_index_name)

        # mount index if it is a loop-device image
        target_index = MINDEX_INFO[prebuilt_index_name]
        if 'image_filesystem' in target_index:
            filesystem = target_index['image_filesystem']
            index_dir = mount_image_index(index_dir, filesystem)

    except ValueError as e:
        print(str(e))
        return None

    print(f'Index directory: {index_dir}')
    return index_dir
