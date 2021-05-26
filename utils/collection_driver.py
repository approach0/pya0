import pya0
import json
from preprocess import tokenize_text


def TREC_preprocess(collection, index, hits):
    if collection in ['test', 'arqmath-2020-task1', 'arqmath-2021-task1', 'arqmath-2021-task1-refined']:
        for hit in hits:
            doc = pya0.index_lookup_doc(index, hit['docid'])
            hit['_'] = hit['docid']
            hit['docid'] = int(doc['url'])

    elif collection in ['arqmath-2020-task2', 'arqmath-2021-task2', 'arqmath-2021-task2-refined']:
        for hit in hits:
            doc = pya0.index_lookup_doc(index, hit['docid'])
            formulaID, postID, threadID, type_, visualID = doc['url'].split(',')
            hit['_'] = formulaID
            hit['docid'] = int(postID)
    else:
        raise NotImplementedError


def TREC_reverse(collection, index, hits):
    if collection in ['test', 'arqmath-2020-task1', 'arqmath-2021-task1', 'arqmath-2021-task1-refined']:
        for hit in hits:
            trec_docid = hit['docid']
            hit['trec_docid'] = trec_docid
            doc = pya0.index_lookup_doc(index, trec_docid)
            hit['docid'] = int(doc['extern_id']) # get internal doc ID
    elif collection in ['arqmath-2020-task2', 'arqmath-2021-task2', 'arqmath-2021-task2-refined']:
        for hit in hits:
            trec_docid = int(hit['_'])
            hit['trec_docid'] = trec_docid
            hit['_'] = str(hit['docid']) # docid is actually post ID here
            doc = pya0.index_lookup_doc(index, trec_docid)
            hit['docid'] = int(doc['extern_id']) # get internal doc ID
    else:
        raise NotImplementedError


def eval_cmd(collection, run_path):
    if collection == 'test':
        return ['sh', 'eval-test.sh', run_path]
    elif collection in ['arqmath-2020-task1', 'arqmath-2021-task1', 'arqmath-2021-task1-refined']:
        return ['sh', 'eval-arqmath-task1.sh', run_path]
    elif collection in ['arqmath-2020-task2', 'arqmath-2021-task2', 'arqmath-2021-task2-refined']:
        return ['sh', 'eval-arqmath-task2.sh', run_path]
    else:
        raise NotImplementedError


def _topic_process__test(idx, line):
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


def _topic_process__ntcir12_math_browsing(idx, line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__ntcir12_math_browsing_concrete(idx, line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__ntcir12_math_browsing_wildcards(idx, line):
    fields = line.split()
    query = [{'type': 'tex', 'str': ' '.join(fields[1:])}]
    qid = fields[0]
    return qid, query, None


def _topic_process__arqmath_2020_task1(idx, json_item):
    query = json_item['kw']
    qid = json_item['qid']
    return qid, query, json_item['tags']


def _topic_process__arqmath_2020_task2(idx, line):
    if line.startswith('#'):
        return None, None, None
    fields = line.split('\t')
    qid = fields[0].strip()
    latex = fields[1].strip()
    treat_type = 'term' if latex.isdigit() or len(latex) == 1 else 'tex'
    query = [{'type': treat_type, 'str': latex}]
    return qid, query, None


def _topic_process__arqmath_2021_task1(idx, line):
    if idx == 0:
        return None, None, None
    fields = line.split('\t')
    qid = fields[0]
    terms = fields[1].strip()
    formulas = fields[2:] if len(fields) > 2 else []
    query_terms = [{'type': 'term', 'str': terms}] if len(terms) > 0 else []
    query_formulas = [{'type': 'tex', 'str': s.strip()} for s in formulas]
    return qid, query_terms + query_formulas, None


def _topic_process__arqmath_2021_task1_refined(idx, line):
    return _topic_process__arqmath_2021_task1(idx, line)


def _topic_process__arqmath_2021_task2(idx, line):
    fields = line.split('\t')
    qid = fields[0]
    formulas = fields[1:]
    query = [{'type': 'tex', 'str': s.strip()} for s in formulas]
    return qid, query, None


def _topic_process__arqmath_2021_task2_refined(idx, line):
    return _topic_process__arqmath_2021_task2(idx, line)


def _featslookup__arqmath_2020_task1(topic_query, index, docid):
    qid, query, qtags = topic_query
    # qnum
    qnum = int(qid.split('.')[1])
    # doc
    doc = pya0.index_lookup_doc(index, docid)
    # doc score
    result_JSON = pya0.search(index, query, verbose=False, topk=1, log=None, docid=docid)
    results = json.loads(result_JSON)
    doc_s = results['hits'][0] if results['ret_code'] == 0 and len(results['hits']) > 0 else {'score': 0}
    score = doc_s['score'] if doc_s['docid'] == docid else 0
    # tags
    dtags = doc['tags']
    qtags = tokenize_text(qtags, no_punctuation=True, rm_stopwords=False)
    dtags = tokenize_text(dtags, no_punctuation=True, rm_stopwords=False)
    n_tag_match = len(set(dtags) & set(qtags))
    # upvotes
    upvotes = int(doc['title'].split(':')[1])
    return [qnum, upvotes, n_tag_match, score]


def _feats_qid_process__arqmath_2020_task1(qfield):
    return 'A.' + qfield.split(':')[1]
