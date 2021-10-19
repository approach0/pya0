import pya0
import json
from preprocess import tokenize_text


def docid_to_doc(index, docid):
    if type(index).__name__ == 'int':
        docid = int(docid)
        doc = pya0.index_lookup_doc(index, docid)
        return doc
    elif isinstance(index, tuple):
        _, docids, _ = index
        return {
            'docid': docid,
            'url': docids[docid][0],
            'content': docids[docid][1]
        }
    elif type(index).__name__ == 'ColBertSearcher':
        docids = index.ext_docIDs
        return {
            'docid': docid,
            'url': docid,
            'content': 'Unindexed'
        }
        quit()
    else:
        raise NotImplementedError


def trec_docid_to_docid(index, trec_docid):
    if type(index).__name__ == 'int':
        trec_docid = int(trec_docid)
        doc = pya0.index_lookup_doc(index, trec_docid)
        docid = int(doc['extern_id'])
        return docid
    else:
        raise NotImplementedError


def TREC_preprocess(collection, index, hits):
    if collection in ['test', 'arqmath-2020-task1', 'arqmath-2021-task1', 'arqmath-2021-task1-refined', 'arqmath-2020-task1-origin', 'arqmath-2021-task1-origin']:
        for hit in hits:
            doc = docid_to_doc(index, hit['docid'])
            hit['_'] = hit['docid'] # save internal docid
            hit['docid'] = int(doc['url']) # output trec docid

    elif collection in ['arqmath-2020-task2', 'arqmath-2021-task2', 'arqmath-2021-task2-refined', 'arqmath-2020-task2-origin', 'arqmath-2021-task2-origin']:
        for hit in hits:
            doc = docid_to_doc(index, hit['docid'])
            formulaID, postID, threadID, type_, visualID = doc['url'].split(',')
            hit['_'] = formulaID # output formula id
            hit['docid'] = int(postID) # output trec docid
    elif collection in ['ntcir12-math-browsing', 'ntcir12-math-browsing-concrete', 'ntcir12-math-browsing-wildcards']:
        for hit in hits:
            doc = docid_to_doc(index, hit['docid'])
            hit['_'] = hit['docid'] # save internal docid
            hit['docid'] = doc['url'] # output trec docid (doc:pos string)
    else:
        raise NotImplementedError


def TREC_reverse(collection, index, hits):
    if collection in ['test', 'arqmath-2020-task1', 'arqmath-2021-task1', 'arqmath-2021-task1-refined']:
        for hit in hits:
            trec_docid = hit['docid']
            hit['trec_docid'] = trec_docid
            try:
                hit['docid'] = trec_docid_to_docid(index, trec_docid)
            except NotImplementedError:
                hit['docid'] = hit['_']
    elif collection in ['arqmath-2020-task2', 'arqmath-2021-task2', 'arqmath-2021-task2-refined']:
        for hit in hits:
            trec_docid = int(hit['_']) # internal (formula) ID
            hit['trec_docid'] = trec_docid
            hit['_'] = str(hit['docid']) # save post ID
            hit['docid'] = trec_docid_to_docid(index, trec_docid)
    elif collection in ['ntcir12-math-browsing', 'ntcir12-math-browsing-concrete', 'ntcir12-math-browsing-wildcards']:
        for hit in hits:
            trec_docid = hit['docid']
            hit['trec_docid'] = trec_docid
            try:
                hit['docid'] = trec_docid_to_docid(index, trec_docid)
            except NotImplementedError:
                hit['docid'] = hit['_']
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


def _topic_process__arqmath_2020_task1_origin(xmlfile):
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    import replace_post_tex
    print(xmlfile)
    for attrs in xmliter(xmlfile, 'Topic'):
        qid = attrs['@number']
        title = attrs['Title']
        post_xml = title + '\n' + attrs['Question']
        s = BeautifulSoup(post_xml, "html.parser")
        post = replace_post_tex.replace_dollar_tex(s.text)
        post = replace_post_tex.replace_alignS_tex(post)
        query = [{'type': 'text', 'str': post}]
        yield qid, query, None


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


def _topic_process__arqmath_2021_task1_origin(xmlfile):
    for qid, query, _ in _topic_process__arqmath_2020_task1_origin(xmlfile):
        yield qid, query, None


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
    doc = docid_to_doc(index, docid)
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
