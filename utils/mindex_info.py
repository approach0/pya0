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
    "arqmath-task1-doclookup": {
        "description": "ARQMath-3 Task-1 corpus document lookup",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/62JME7BSqXex7me/download"
        ],
        "md5": "84213bcff54e61c86e4d556bc4ebfde4"
    },
    "arqmath-task1-doclookup-full": {
        "description": "ARQMath-3 Task-1 corpus document lookup (w/ questions)",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/FYNyHkGC4xkZnj4/download"
        ],
        "md5": "b2916e2bfea9bcec195348ea6a8f9fcc"
    },
    "MATH-dpr-cocomae-220-hnsw": {
        "description": "MATH index encoded by Coco-MAE 220",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/N8atJR3Kqf7Aoni/download"
        ],
        "md5": "980a8ba9b7269e37fbad11876abee1fb"
    },
    "MATH-dpr-cocomae-520-hnsw": {
        "description": "MATH index encoded by Coco-MAE 520",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/n8cjmFKo7HGwQig/download"
        ],
        "md5": "8bb6602db36a30e4bd325de3e95e49f7"
    },
    "MATH-unsup": {
        "description": "MATH dataset solutions indexed by approach0",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/fWZ86BFdjQFT4ik/download"
        ],
        "md5": "c98f78d3d7ef92f8ffe3915ff0b14f13",
        "image_filesystem": "reiserfs"
    },
    "arqmath-duplicate-questions": {
        "description": "ARQMath duplicate questions index (See https://huggingface.co/datasets/approach0/MSE-duplicate-questions)",
        "urls": [
            "https://vault.cs.uwaterloo.ca/s/Y4JXcdnoRZQNZYi/download"
        ],
        "md5": "59bce646b0959aa60864c046247bd4e9",
        "image_filesystem": "reiserfs"
    },
}

def list_indexes():
    df = pd.DataFrame.from_dict(MINDEX_INFO)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
        'display.max_colwidth', None, 'display.colheader_justify', 'left'):
        print(df)
