cfg_path = '.training-and-inference/inference.ini'
MATH_path = '/home/w32zhong/msr/datasets/MATH/test/precalculus'
default_tokenizer = 'approach0/dpr-cocomae-220'
single_vec_model = 'approach0/dpr-cocomae-220'
prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'

import os
import json
import sys
sys.path.insert(0, '.')

from pya0.index_manager import from_prebuilt_index
from pya0.transformer_eval import search
from pya0.replace_post_tex import replace_dollar_tex, replace_display_tex, replace_inline_tex
from pya0.transformer_eval import psg_encoder__dpr_default, searcher__docid_vec_flat_faiss

print('Loading model...')
index_path = from_prebuilt_index(prebuilt_index)
encoder, enc_utils = psg_encoder__dpr_default(default_tokenizer, single_vec_model, 0, 0, 'cpu')
searcher, _ = searcher__docid_vec_flat_faiss(index_path, None, enc_utils, 'cpu')


for filename in os.listdir(MATH_path):
    json_path = os.path.join(MATH_path, filename)
    with open(json_path, 'r') as fh:
        j = json.load(fh)
    query = j['problem']
    query = replace_dollar_tex(query)
    query = replace_display_tex(query)
    query = replace_inline_tex(query)
    results = searcher(query, encoder, topk=5, debug=False)

    print('TEST:', json_path)
    print('Q:', j['problem'], end='\n\n')
    print('A:', j['solution'], end='\n\n')
    for i, res in enumerate(results):
        d = res[2][1]
        d = d.replace(r'[imath]', '$')
        d = d.replace(r'[/imath]', '$')
        print(f'D({1+i}):', d, end='\n\n')

    input('Hit Enter for the next...')
