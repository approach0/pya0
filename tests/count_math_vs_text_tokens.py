import sys
sys.path.insert(0, '.')
from pya0.preprocess import *
from pya0.transformer_eval import *

text_toks, math_toks = 0, 0
text_length, math_length = 0, 0
for _, query in corpus_reader__flat_topics('arqmath-2022-task1-or-task3-origin'):
    for type_, piece, *_ in iter_imath_splits(query):
        if type_ == 'math':
            tokens = tex_tokenize(piece, include_spans=True, include_syntatic_literal=True)
            math_toks += len(tokens)
            math_length += len(piece)
        elif type_ == 'text':
            tokens = tokenize_text(piece, no_punctuation=True)
            text_toks += len(tokens)
            text_length += len(piece)
print(math_toks / (text_toks + math_toks))
print(math_length / (text_length + math_length))
