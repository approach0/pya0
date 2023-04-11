import pandas as pd

MINDEX_INFO = {
    "ntcir-wfb": {
        "description": "NTCIR-12 Wikipedia Formula Browsing",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/gySLti89gZF8xz6/download"
        ],
        "md5": "6e87fc52a8f02c05113034c4b14b3e06",
        "image_filesystem": "reiserfs"
    },
    "arqmath-task1": {
        "description": "ARQMath-3 Task-1 corpus index",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/c79mXwi8kkAMPSw/download"
        ],
        "md5": "24cd1b4a302ec3f5c0ae15508d3c79f7",
        "image_filesystem": "reiserfs"
    },
    "arqmath-task2": {
        "description": "ARQMath-3 Task-2 corpus index",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/zCtBtR7Y93fpBby/download"
        ],
        "md5": "f0256f6e7fa7b87bc855e4b10050ba91",
        "image_filesystem": "reiserfs"
    }
}

def list_indexes():
    df = pd.DataFrame.from_dict(MINDEX_INFO)
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', -1, 'display.colheader_justify', 'left'):
        print(df)
