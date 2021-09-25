from collections import Counter
from preprocess import tokenize_query, tokenize_content
import pya0
import re


def list2vec(lst):
    return dict(Counter(lst))


def normalize_vec(vec):
    norm = sum(vec.values())
    for key in vec:
        vec[key] /= norm


def interpolate(vec_q, vec_d, alpha=0.5):
    combined_vocab = set()
    combined_vocab.update(vec_q.keys())
    combined_vocab.update(vec_d.keys())
    vec_new = {}
    for w in combined_vocab:
        wq = vec_q[w] if w in vec_q else 0
        wd = vec_d[w] if w in vec_d else 0
        vec_new[w] = alpha * wq + (1 - alpha) * wd
    return vec_new


def rm3_expand_query(index, query, hits, feedbackTerms=20, feedbackDocs=10):
    # create hit vectors
    q_lst = tokenize_query(query)
    vocab = set()
    d_vectors = []
    d_scores = []
    for hit in hits[:feedbackDocs]:
        docID = hit['docid']
        doc = pya0.index_lookup_doc(index, docID)
        d_lst = tokenize_content(doc['content'], whitelist=q_lst)
        d_vec = list2vec(d_lst)
        d_vectors.append(d_vec)
        d_scores.append(hit['score'])
        vocab.update(d_vec.keys())

    # generate relevance_model (RM)
    relevance_model = dict([(w, 0) for w in vocab]) # P(w|R) \prox P(w|q1...qn) / Z
    for word in vocab:
        word_weight = 0 # P(w,q1...qn) = sum_D P(D) * P(w|D) * QueryLikelihood
        for i, d_vec in enumerate(d_vectors):
            freq = d_vec[word] if word in d_vec else 0
            score = d_scores[i]
            norm = sum(d_vec.values())
            word_weight += (freq / norm) * score
        relevance_model[word] = word_weight

    # P(w|R) \prox P(w,q1...qn) / sum_w P(w,q1...qn)
    normalize_vec(relevance_model)

    # query RM normalization (L1)
    q_vec = list2vec(q_lst)
    normalize_vec(q_vec)

    # interpolate document RM with query RM
    relevance_model = interpolate(q_vec, relevance_model)

    # sort and select the top feedback terms as new query keywords
    new_query = sorted(relevance_model.items(), key=lambda item: item[1], reverse=True)
    new_query = new_query[:max(feedbackTerms, len(query))]
    new_query = dict(new_query)
    normalize_vec(new_query)

    # convert to required query format
    query = []
    for q in new_query:
        if q.find('[imath]') >= 0:
            splits = re.split('\[imath\]|\[/imath\]', q)
            query.append({
                'type': 'tex',
                'str': splits[1],
                'field': 'content',
                'boost': new_query[q]
            })
        else:
            query.append({
                'type': 'term',
                'str': q,
                'field': 'content',
                'boost': new_query[q]
            })
    return query
