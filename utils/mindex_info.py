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
    },
    "arqmath-task1-dpr-cocomae-220-hnsw": {
        "description": "ARQMath-3 Task-1 efficient dense index",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/DGyWWFMFWQ4YSz2/download"
        ],
        "md5": "159440c22672ca2f462fdb7d8ac9cad6"
    },
    "arqmath-task1-dpr-cocomae-220": {
        "description": "ARQMath-3 Task-1 flat dense index",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/XkkgRzpgWjYtLmc/download"
        ],
        "md5": "7f2a69e6374bc152df9f13b8c7a5aebb"
    },
    "arqmath-task1-colbert-cocomae-600": {
        "description": "ARQMath-3 Task-1 multi-vector dense index",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/sWTN9NS4mM7fA4P/download"
        ],
        "md5": "89acb126a8d3706b4c4e8d3329a108eb"
    },
}

def list_indexes():
    df = pd.DataFrame.from_dict(MINDEX_INFO)
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', -1, 'display.colheader_justify', 'left'):
        print(df)
