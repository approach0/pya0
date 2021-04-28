import pya0
from preprocess import tokenize_text


def TREC_preprocess(collection, index, hits):
    if collection == 'arqmath-2020-task1' or collection == 'test':
        for hit in hits:
            doc = pya0.index_lookup_doc(index, hit['docid'])
            hit['_'] = hit['docid']
            hit['docid'] = int(doc['url'])

    elif collection == 'arqmath-2020-task2':
        for hit in hits:
            doc = pya0.index_lookup_doc(index, hit['docid'])
            formulaID, postID, threadID, type_, visualID = doc['url'].split(',')
            hit['_'] = formulaID
            hit['docid'] = int(postID)


def TREC_reverse(collection, index, hits):
    if collection == 'arqmath-2020-task1' or collection == 'test':
        for hit in hits:
            trec_docid = hit['docid']
            doc = pya0.index_lookup_doc(index, trec_docid)
            hit['docid'] = int(doc['extern_id'])


def eval_cmd(collection, run_path):
    if collection == 'test':
        return ['sh', 'eval-test.sh', run_path]
    elif collection == 'arqmath-2020-task1':
        return ['sh', 'eval-arqmath-task1.sh', run_path]
    elif collection == 'arqmath-2020-task2':
        return ['sh', 'eval-arqmath-task2.sh', run_path]
    else:
        return None


def _topic_process__test(line):
    fields = line.split(',')
    qid = fields[0]
    tags = fields[1]
    keywords = fields[2:]
    query = []
    for kw in keywords:
        kw = kw.strip()
        type_ = 'tex' if kw[0] == '$' else 'term'
        str_ = kw.strip('$') if kw[0] == '$' else kw
        query.append({
            'type': type_,
            'str': str_
        })
    return qid, query, tags


def _topic_process__ntcir12_math_browsing(line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__ntcir12_math_browsing_concrete(line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__ntcir12_math_browsing_wildcards(line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__arqmath_2020_task1(json_item):
    query = json_item['kw']
    qid = json_item['qid']
    return qid, query, json_item['tags']


def _topic_process__arqmath_2020_task2(line):
    if line.startswith('#'):
        return None, None, None
    fields = line.split('\t')
    qid = fields[0].strip()
    latex = fields[1].strip()
    treat_type = 'term' if latex.isdigit() or len(latex) == 1 else 'tex'
    query = [{'type': treat_type, 'str': latex}]
    return qid, query, None


def _featslookup__test(topic_query, doc):
    qid, query, qtags = topic_query
    # qnum
    qnum = int(qid.split('.')[1])
    # tags
    dtags = doc['tags']
    qtags = tokenize_text(qtags, no_punctuation=True, rm_stopwords=False)
    dtags = tokenize_text(dtags, no_punctuation=True, rm_stopwords=False)
    n_tag_match = len(set(dtags) & set(qtags))
    # upvotes
    upvotes = int(doc['title'].split(':')[1])
    return [qnum, upvotes, n_tag_match]


def _featslookup__arqmath_2020_task1(topic_query, doc):
    qid, query, qtags = topic_query
    # qnum
    qnum = int(qid.split('.')[1])
    # tags
    dtags = doc['tags']
    qtags = tokenize_text(qtags, no_punctuation=True, rm_stopwords=False)
    dtags = tokenize_text(dtags, no_punctuation=True, rm_stopwords=False)
    n_tag_match = len(set(dtags) & set(qtags))
    # upvotes
    upvotes = int(doc['title'].split(':')[1])
    return [qnum, upvotes, n_tag_match]
