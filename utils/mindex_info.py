import pandas as pd

MINDEX_INFO = {
    "ntcir-wfb": {
        "description": "NTCIR-12 Wikipedia Formula Browsing",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/gySLti89gZF8xz6/download"
        ],
        "md5": "6e87fc52a8f02c05113034c4b14b3e06",
        "image_filesystem": "reiserfs"
    }
}

def list_indexes():
    df = pd.DataFrame.from_dict(MINDEX_INFO)
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', -1, 'display.colheader_justify', 'left'):
        print(df)
