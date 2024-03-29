import re
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from _pya0 import tokenize as tex_tokenize

stemmer_func = lambda x: x
tokenizer = RegexpTokenizer(r'\w+')
detokenizer = TreebankWordDetokenizer()


def use_stemmer(name='lancaster'):
    global stemmer_func
    if name is None:
        stemmer_func = lambda x: x
    elif name == 'lancaster':
        stemmer_func = LancasterStemmer().stem
    elif name == 'porter':
        stemmer_func = PorterStemmer().stem
    else:
        raise NotImplementedError


def preprocess_text(txt):
    if len(txt) == 0:
        return txt
    lpad = ' ' if txt[0] == ' ' else ''
    rpad = ' ' if txt[-1] == ' ' else ''
    stem_toks = tokenize_text(txt)
    txt = detokenizer.detokenize(stem_toks)
    return f'{lpad}{txt}{rpad}'


def tokenize_text(txt, no_punctuation=False, rm_stopwords=False, whitelist=[]):
    toks = tokenizer.tokenize(txt) if no_punctuation else word_tokenize(txt)
    stem_toks = []
    my_stopwords = (
        stopwords.words('english') +
        ['__EXPAND__', '__answer__']
    )
    for t in toks:
        if rm_stopwords and t in my_stopwords and t not in whitelist:
            continue
        tt = t.split('-')
        t = '-'.join([stemmer_func(x) for x in tt])
        stem_toks.append(t)
    return stem_toks


def expand_math(tex, expand_terms):
    for _, tok, sym in tex_tokenize(tex):
        if tok == 'VAR' and sym.find('`') == -1:
            expand_terms.add(sym)
        elif tok == 'VAR' and sym == "blackboard-bold`R'":
            expand_terms.add('rational')
            expand_terms.add('number')
        elif tok == 'VAR' and sym == "blackboard-bold`N'":
            expand_terms.add('natural')
            expand_terms.add('number')
        elif tok == 'PI':
            expand_terms.add('pi')
        elif tok == 'ZERO':
            expand_terms.add('zero')
        elif tok == 'INFTY':
            expand_terms.add('infinity')
        elif sym in ('equal', 'neq'):
            expand_terms.add('equality')
        elif sym in ('gt', 'lt', 'le', 'ge'):
            expand_terms.add('inequality')
        elif sym in ('int', 'oint'):
            expand_terms.add('integral')
        elif sym in ('sum'):
            expand_terms.add('summation')
        elif sym == 'frac':
            expand_terms.add('fraction')
        elif sym == 'root':
            expand_terms.add('root')
        elif tok == 'PARTIAL':
            expand_terms.add('partial')
            expand_terms.add('derivative')
        elif tok == 'FACT':
            expand_terms.add('factorial')
        elif tok == 'MODULAR':
            expand_terms.add('modular')
            expand_terms.add('mod')
        elif tok == 'NAME_FUN':
            expand_terms.add(sym)
        elif tok == 'TRIGONOMETRIC':
            expand_terms.add(sym)
            if sym == 'sin':
                expand_terms.add('sine')
            elif sym == 'cos':
                expand_terms.add('cosine')
            elif sym == 'tan':
                expand_terms.add('tangent')


def iter_imath_splits(content):
    last = None
    splits = re.split('(\[imath\]|\[/imath\])', content)
    for i, cur in enumerate(splits):
        next_ = splits[i + 1] if i + 1 < len(splits) else None
        trim = lambda x: None if x is None else x.strip()
        if cur.strip() == '':
            continue
        elif trim(cur) in ('[imath]', '[/imath]'):
            last = cur
            continue
        elif trim(last) == None and trim(next_) == '[imath]':
            yield ('text', cur, None, None)
        elif trim(next_) == None and trim(last) == '[/imath]':
            yield ('text', cur, None, None)
        elif trim(last) == '[imath]' and trim(next_) == '[/imath]':
            yield ('math', cur, last, next_)
        else:
            yield ('text', cur, None, None)
        last = cur


def preprocess(content, expansion=False):
    output = ''
    expand_set = set()
    for type_, piece, last, next_ in iter_imath_splits(content):
        # print(type_, piece, last, next_)
        if type_ == 'text':
            output += preprocess_text(piece)
        else:
            if expansion: expand_math(piece, expand_set)
            output += last + piece + next_
    if expansion:
        expands = ' '.join(list(expand_set))
        output += ' __EXPAND__ '
        for type_, piece, last, next_ in iter_imath_splits(expands):
            if type_ == 'text': output += preprocess_text(piece)
    return output.strip()


def preprocess_query(query, expansion=False, query_type_filter=None):
    # type filter
    if query_type_filter is not None:
        query = filter(lambda kw: kw['type'] == query_type_filter, query)
        query = list(query)
    # math expansion
    expand_set = set()
    for kw in query:
        if kw['type'] == 'tex':
            if expansion: expand_math(kw['str'], expand_set)
    if expansion:
        for expand_term in expand_set:
            query.append({
                'type': 'term',
                'str': expand_term,
            })
    # query text keyword process
    for kw in query:
        if kw['type'] == 'term':
            kw['str'] = preprocess_text(kw['str'])
        kw['field'] = 'content'
    return query


def tokenize_content(content, whitelist=[]):
    tokens = []
    for type_, piece, last, next_ in iter_imath_splits(content):
        if type_ == 'text':
            toks = tokenize_text(piece, no_punctuation=True,
                                 rm_stopwords=True, whitelist=whitelist)
            tokens = tokens + toks
        else:
            tok = last + piece + next_
            tokens.append(tok)
    return tokens


def tokenize_content_by_sentence(content):
    sentences = ['.']
    for type_, piece, last, next_ in iter_imath_splits(content):
        if type_ == 'text':
            for sent in sent_tokenize(piece):
                last_sent = sentences[-1].strip()
                if last_sent[-1] in [';', '.']:
                    sentences.append(sent)
                else:
                    sentences[-1] += ' ' + sent
        else:
            math_piece = last + piece + next_
            last_sent = sentences[-1].strip()
            if last_sent[-1] in [';', '.']:
                sentences.append(math_piece)
            else:
                sentences[-1] += ' ' + math_piece
    sentences.pop(0)
    return sentences


def tokenize_query(query):
    tokens = []
    for q in query:
        kw = q['str']
        type_ = q['type']
        if type_ == 'term':
            toks = tokenize_text(kw, no_punctuation=True)
            tokens += toks
        elif type_ == 'tex':
            tokens.append(f'[imath]{kw}[/imath]')
        else:
            raise NotImplementedError

    return tokens


def unwrap_isolated_tex_group(text, group_name):
    regex = re.compile(
        r"\\begin{" + group_name +
        r"\*?}(.+?)\\end{" + group_name +
        r"\*?}(?!\s*\[/imath\])", re.DOTALL) # negative lookahead
    return re.sub(regex, r"[imath]\1[/imath]", text)


def unwrap_isolated_tex_groups(text,
    groups=['align', 'alignat', 'equation', 'gather']):
    for grp in groups:
        text = unwrap_isolated_tex_group(text, grp)
    return text


def preprocess_for_transformer(text, math_vocab=None, text_lowercase=True,
    num_tokenizer_ver=3, replace_isolated_groups=True, dest_token='pya0'):

    if dest_token == 'math_albert':
        text = text.replace('[imath]', '$')
        text = text.replace('[/imath]', '$')
        return text

    output = ''
    if replace_isolated_groups:
        text = unwrap_isolated_tex_groups(text)

    def num_tokenizer_v1(piece, tok_type, sym, span):
        if '`' in sym:
            sym = sym.split('`')[-1].strip('\'')
        if tok_type == 'NUM' and len(str(sym)) >= 2:
            sym = 'somenum'
        return [sym]
    def num_tokenizer_v2(piece, tok_type, sym, span):
        number_str = piece[slice(*span)]
        return list(number_str) # buggy!
    def num_tokenizer_v3(piece, tok_type, sym, span):
        #print(tok_type, piece, sym)
        if tok_type == 'VAR':
            if '`' in sym:
                string = sym.split('`')[-1].strip('\'')
            else:
                string = sym.strip('\\')
            return [string]
        else:
            string = piece[slice(*span)]
            return list(string)
    num_tokenizer = locals()['num_tokenizer_v' + str(num_tokenizer_ver)]

    for type_, piece, *_ in iter_imath_splits(text):
        if type_ == 'math':
            tex_toks = []
            try:
                tex_toks = tex_tokenize(piece, include_spans=True,
                    include_syntatic_literal=True)
            except Exception as err:
                print(err)
                print('Occurred when parsing:', piece.strip('\n'))
                continue
            tex_syms = []
            for _, tok_type, sym, span in tex_toks:
                if tok_type in ('VAR', 'NUM', 'FLOAT', 'ONE', 'ZERO'):
                    split_syms = num_tokenizer(piece, tok_type, sym, span)
                elif sym == '\n':
                    break
                else:
                    split_syms = [sym]
                for sym in split_syms:
                    dollar_prefix_sym = '$' + sym + '$'
                    tex_syms.append(dollar_prefix_sym)
                    if math_vocab is not None:
                        math_vocab[dollar_prefix_sym] += 1
            output += ' '.join(tex_syms)
        else:
            output += piece.lower() if text_lowercase else piece
    return output


if __name__ == '__main__':
    #math = r'4.7 = 1+\frac{1+2}{2!}+\frac{1+2+3}{3!}+\cdots+\frac{1+2+3+...+20}{20!}'
    #math = r'1.50941045653627123243833773286186'
    #math = r'αK'
    math = r'''
How to express these equations as polynomial equations in [imath]x,
\alpha, z[/imath]?
\begin{align*}
x&=\cos(2\alpha)\\
z&=\sin(2\alpha)
\end{align*}
I expect that the polynomial equations can give the same surface in [imath]\mathbb R^3[/imath]:
[imath]
x^2+y^2+z^2-4a^2+4b^2-5=0
[/imath]
    '''
    output = preprocess_for_transformer(math, dest_token='math_albert')
    print(output)
