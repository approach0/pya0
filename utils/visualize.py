import json
import os
from functools import partial
from transformer_eval import auto_invoke, gen_flat_topics
from eval import gen_topics_queries
from eval import get_qrels_filepath, parse_qrel_file
from mergerun import parse_trec_file, parse_task3_file
from preprocess import preprocess_query
import collection_driver


def parse_scores_file(file_path):
    scores_per_topic = dict()
    if not os.path.exists(file_path):
        return scores_per_topic
    with open(file_path, 'r') as fh:
        for line in fh.readlines():
            line = line.rstrip()
            fields = line.split('\t')
            metric = fields[0].strip()
            qryID = fields[1]
            score = fields[2]
            if qryID not in scores_per_topic:
                scores_per_topic[qryID] = dict()
            scores_per_topic[qryID][metric] = score
    return scores_per_topic


def output_html_pagination(fh, qid, page, tot_pages):
    fh.write('<div style="margin: 1rem 0;">\n')
    fh.write(f'[<a href="/">home</a> ]')
    fh.write(f'[<a href=".">up</a> ]')
    if page > 0:
        fh.write(f'[<a href="{qid}__p{page - 0:03}.html">&lt;&lt; prev</a> ]')
    if page + 1 < tot_pages:
        fh.write(f'[<a href="{qid}__p{page + 2:03}.html">next &gt;&gt;</a> ]')
    fh.write('</div>\n')


def degree_color(relev):
    if relev == -1:
        return 'white'
    elif relev == 0:
        return 'darkgray'
    elif relev == 1:
        return 'palegoldenrod'
    elif relev == 2:
        return '#FFEB3B'
    elif relev == 3:
        return 'gold'
    elif relev == 4:
        return 'goldenrod'
    else:
        return 'red'


def output_html(output_dir, output_name, qid, query, hits, qrels,
    judged_only, hits_per_page, generator_mapper):
    # prepare output
    hits_per_page = 100
    tot_pages = len(hits) // hits_per_page + (len(hits) % hits_per_page > 0)
    parent_dir = f'{output_dir}/{output_name}' + ('__J' if judged_only else '')
    parent_dir = os.path.expanduser(parent_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    for page in range(tot_pages):
        start = page * hits_per_page
        page_hits = hits[start : start + hits_per_page]
        # start output page
        with open(f'{parent_dir}/{qid}__p{page + 1:03}.html', 'w') as fh:
            mathjax_cdn = "https://cdn.jsdelivr.net/npm/mathjax@3.2.0/es5/tex-chtml-full.js"
            fh.write('<html>\n')
            # head
            fh.write('<head>\n')
            fh.write(f'<title>{qid} (page #{page + 1})</title><body>\n')
            fh.write('<style>\n')
            fh.write('a, a:visited { color: blue; } \n')
            fh.write('#topbar { position: sticky; top: 0; z-index: 999; ' +
                     ' background: white; border-bottom: grey solid 1px; } \n')
            fh.write('</style>\n')
            fh.write('</head>\n')
            # query
            fh.write(f'<h3>Query Keywords (Topic ID: {qid})</h3>\n')
            fh.write('<ul id="topbar">\n')
            for q in query:
                if q['type'] == 'term':
                    kw_str = f'{q["str"]} &nbsp;&nbsp;'
                else:
                    kw_str = f'[imath]{q["str"]}[/imath]'
                fh.write(f'<li>{kw_str}</li>\n')
            fh.write('</ul>\n')
            # hits
            fh.write(f'<h3>Hits (page #{page + 1} / {tot_pages})</h3>\n')
            output_html_pagination(fh, qid, page, tot_pages)
            fh.write('<ol>\n')
            for hit in page_hits:
                docID = hit["trec_docid"]
                rank = hit['rank']
                score = hit['score']
                relev = hit['relev']
                content = hit['content'].replace(r'\require', '')
                fh.write('<li>\n')
                fh.write(f'<b id="{rank}"><a href="#{rank}">#{rank}</a>, ' +
                         f'doc#{docID}, score: {score}, relev: {relev}, </b>\n')
                colors = [
                    f'<b style="background: {degree_color(i)}">{i}</b>'
                    for i in range(4)
                ]
                color = degree_color(relev)
                fh.write('<b>relevance levels: ' + ' '.join(colors) + ':</b>')
                fh.write(f'<p style="background: {color};">{content}</p>\n')
                if generator_mapper is not None:
                    generator_mapper(fh, query, hit)
                fh.write('</li>\n')
            fh.write(f'</ol>\n')
            output_html_pagination(fh, qid, page, tot_pages)
            # mathJax
            fh.write('<script> window.MathJax ={' +
                "loader: { source: {'[tex]/AMScd': '[tex]/amscd'} }," +
                'tex: { inlineMath: [ ["$","$"], ["[imath]", "[/imath]"] ] }' +
            '}; </script>')
            fh.write(f'<script src="{mathjax_cdn}"></script>')
            # end document
            fh.write('</body></html>\n')


def generator__colbert(tokenizer_path, model_path, config):
    import uuid
    from transformer_utils import colbert_init, colbert_infer
    use_puct_mask = config.getboolean('use_puct_mask')
    model, tokenizer, prepends = colbert_init(
        model_path, tokenizer_path, use_puct_mask=use_puct_mask)

    def map_degree(score):
        if score >= 0.97:
            return degree_color(4)
        elif score >= 0.96:
            return degree_color(3)
        elif score >= 0.95:
            return degree_color(2)
        elif score >= 0.92:
            return degree_color(1)
        else:
            return degree_color(0)

    def mapper(fh, query, hit):
        Q = query[0]['str']
        D = hit['content']
        infer_score, cmp_matrix, (enc_Q, enc_D) = colbert_infer(
            model, tokenizer, prepends, Q, D,  q_augment=False
        )
        tok_Q = [
            tokenizer.decode(x).replace(' ', '') for x in enc_Q
        ]
        tok_D = [
            tokenizer.decode(x).replace(' ', '') for x in enc_D
        ]
        q_weights = cmp_matrix.max(1).tolist()
        q_indexes = cmp_matrix.argmax(1).tolist()
        idx_Q = list(zip(tok_Q, q_weights, q_indexes))

        fh.write(f'<b>colbert score: {infer_score:.5f}</b>')
        # write query
        fh.write('<p>')
        for i, (q_kw, score, index) in enumerate(idx_Q):
            uid = uuid.uuid4().hex.upper()[0:8] + '-' + str(i)
            if q_kw.startswith('$'):
                q_kw = q_kw.strip('$')
                q_kw = f'（{q_kw}）'
            color = map_degree(score)
            tok_D[index] = (tok_D[index], color, uid)
            fh.write(
                f'''
                <span style="background-color:{color}"
                 onmouseenter="
                    //console.log('{uid}');
                    let ele = document.getElementsByClassName('{uid}')[0]
                    ele.style.color = 'cyan';
                 "
                 onmouseleave="
                    let ele = document.getElementsByClassName('{uid}')[0]
                    ele.style.color = 'black';
                 "
                 >
                    {q_kw}
                </span>
                '''
            )
        fh.write('</p>')
        # write document
        fh.write('<p>')
        for d in tok_D:
            linked = []
            while isinstance(d, tuple):
                d, color, uid = d
                linked.append(uid)
            if d.startswith('$'):
                d = d.strip('$')
                d = f'（{d}）'
            if len(linked):
                fh.write(f'<span style="background-color:{color}" class="')
                for l in linked:
                    fh.write(l + ' ')
                fh.write(f'"> {d} </span>')
            else:
                fh.write(f'{d} ')
        fh.write('</p>')

    return mapper


def generator__splade(tokenizer_path, model_path, dim, config):
    import torch
    import uuid
    from collections import defaultdict
    from transformers import BertTokenizer
    from transformer import SpladeMaxEncoder
    from pya0.preprocess import preprocess_for_transformer

    vocab_topk = config.getint('vocab_topk')
    model = SpladeMaxEncoder.from_pretrained(model_path,
        tie_word_embeddings=True)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    offset = len(vocab) - dim

    def map_degree(score):
        if score >= 1.5:
            return degree_color(4)
        elif score >= 1.0:
            return degree_color(3)
        elif score >= 0.5:
            return degree_color(2)
        elif score >= 0.2:
            return degree_color(1)
        else:
            return degree_color(0)

    def output_vocab_vec(fh, vec, zip_Q):
        uid = uuid.uuid4().hex.upper()[0:8]
        maps = defaultdict(list)
        # write tokens
        fh.write('<p>')
        for i, (tok, tok_vec) in enumerate(zip_Q):
            tok_id = vocab[tok]
            tok_score = tok_vec[tok_id]
            tok_uid = uid + '-' + str(i)
            for i in range(dim):
                if tok_vec[offset + i] > 0:
                    maps[i].append(tok_uid)
            if tok.startswith('$'):
                tok = tok.strip('$')
                tok = f'（{tok}）'
            color = map_degree(tok_score)
            fh.write(
                f'''
                <span style="background-color:{color}"
                 onmouseenter="
                    let eles = document.getElementsByClassName('{tok_uid}')
                    for (const ele of eles) {{
                        ele.style.borderColor = 'cyan';
                    }}
                 "
                 onmouseleave="
                    let eles = document.getElementsByClassName('{tok_uid}')
                    for (const ele of eles) {{
                        ele.style.borderColor = 'black';
                    }}
                 "
                 >
                    {tok}
                </span>
                '''
            )
        fh.write('</p>')
        # write vector grids
        fh.write('<p><div style="width: 900px">')
        for i in range(dim):
            if vec[0][offset + i] > 0:
                classes = ' '.join(maps[i])
                fh.write(
                    f'''
                    <div id="{uid}-vocab-{offset + i}" class="{classes}"
                     title="<{inv_vocab[offset + i]}> {vec[0][offset + i]}@{i}"
                     style="display: inline-block; cursor: pointer;
                     background: {map_degree(vec[0][offset + i])};
                     width: 4px; height: 4px; border: grey solid 1px;">
                    </div>
                    '''
                )
        fh.write('</div></p>')

    def mapper(fh, query, hit):
        Q = preprocess_for_transformer(query[0]['str'])
        D = preprocess_for_transformer(hit['content'])
        enc_Q = tokenizer(Q, truncation=True, return_tensors="pt")
        enc_D = tokenizer(D, truncation=True, return_tensors="pt")
        tok_Q = [
            tokenizer.decode(x).replace(' ', '') for x in enc_Q['input_ids'][0]
        ]
        tok_D = [
            tokenizer.decode(x).replace(' ', '') for x in enc_D['input_ids'][0]
        ]
        with torch.no_grad():
            out_Q, out_D = model(enc_Q), model(enc_D)
            zip_Q = zip(tok_Q, out_Q[2][0].tolist())
            zip_D = zip(tok_D, out_D[2][0].tolist())
            vec_q = out_Q[1][0][offset:]
            vec_d = out_D[1][0][offset:]
            overall_score = vec_q.T @ vec_d
            top_eles = (vec_q * vec_d).topk(vocab_topk)

        fh.write(f'<b>splade score: {overall_score:.5f}</b><br/>')
        fh.write(f'<b>top dim:</b> ')
        for partial_score, idx in zip(
            top_eles.values.tolist(), top_eles.indices.tolist()):
            tok = inv_vocab[offset + idx]
            if tok.startswith('$'):
                tok = tok.strip('$')
                tok = f'（{tok}）'
            fh.write(f'{tok} {partial_score:.5f} | ')
        output_vocab_vec(fh, out_Q[1], zip_Q)
        output_vocab_vec(fh, out_D[1], zip_D)

    return mapper


def visualize_file(config_file, section, input_file):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)

    # parse common config
    topk = config.getint('DEFAULT', 'topk')
    judged_only = config.getboolean('DEFAULT', 'judged_only')
    output_name = os.path.basename(input_file)
    filter_topics = json.loads(config['DEFAULT']['filter_topics'])
    hits_per_page = config.getint('DEFAULT', 'hits_per_page')
    output_dir = config['DEFAULT']['output_dir']
    print('filter_topics:', filter_topics)

    # parse lookup index config
    index_path = config['DEFAULT']['lookup']
    if ':' in index_path:
        index = collection_driver.open_special_index(index_path)
    else:
        raise NotImplementedError

    # parse input relevant configs
    input_type = config[section]['input_type']
    if input_type == 'trec_runfile':
        hits_per_topic, _ = parse_trec_file(input_file)
    else:
        raise NotImplementedError

    # parse topic relevant configs
    topic_type, topic_arg = json.loads(config[section]['topics'])
    if topic_type == 'topics':
        collection = topic_arg
        # filter out "None line" or header line
        gen_topic = filter(lambda x: x[0], gen_topics_queries(collection))
        gen_topic = [(qid, preprocess_query(q)) for qid, q, _ in gen_topic]
        qrels_path = get_qrels_filepath(collection)
    elif topic_type == 'flat':
        collection = topic_arg
        gen_topic = list(gen_flat_topics(collection, 'space'))
        assert isinstance(gen_topic[0][1], str)
        gen_topic = [(qid, [{'type': 'term', 'str': q}])
            for qid, q in gen_topic]
        qrels_path = get_qrels_filepath(collection)
    else:
        raise NotImplementedError
    print('topic:', topic_arg)
    print('QRELS:', qrels_path)
    qrels = parse_qrel_file(qrels_path) if qrels_path else {}

    # parse generator relevant configs
    if 'generator' in config[section]:
        print(config[section]['generator'])
        generator_mapper = auto_invoke(
            'generator', config[section]['generator'],
            extra_args=[config[section]], global_ids=globals()
        )
    else:
        generator_mapper = None

    for qid, query in gen_topic:
        if len(filter_topics) > 0 and qid not in filter_topics:
            continue
        print(qid, query)
        topic_hits = hits_per_topic[qid] if qid in hits_per_topic else []
        # fix ranks based on score (just like trec_eval does)
        topic_hits = sorted(topic_hits,
            key=lambda x: (x['score'], x['docid']), reverse=True)
        # preprocess hits
        collection_driver.TREC_reverse(collection, index, topic_hits)
        for j, hit in enumerate(topic_hits):
            # set fixed rank
            hit['rank'] = j + 1
            # set content
            int_docid = hit['docid'] # must be internal docid
            doc = collection_driver.docid_to_doc(index, int_docid)
            hit['content'] = doc['content']
            # set relevance
            qrel_id = f'{qid}/' + hit["trec_docid"]
            relev = int(float(qrels[qrel_id])) if qrel_id in qrels else -1
            hit['relev'] = relev
        # filter hits
        if judged_only:
            topic_hits = [h for h in topic_hits if h['relev'] > -1]
        topic_hits = topic_hits[:topk]
        # output this topic
        output_html(output_dir, output_name, qid, query, topic_hits, qrels,
            judged_only, hits_per_page, generator_mapper)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1.3, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    import matplotlib.pyplot as plt

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def visualize_score_files_in_bar_graph(score_files):
    import re
    all_topics = set()
    for scores_file_path in score_files:
        scores = parse_scores_file(scores_file_path)
        all_topics.update(scores.keys())
    def numerically(x):
        m = re.search(r'\d+', x.__str__())
        return 0 if m is None else int(m.group())
    all_topics = sorted(all_topics, key=numerically)
    import matplotlib.pyplot as plt

    data = dict()
    for scores_file_path in score_files:
        name = os.path.basename(scores_file_path)
        if name == '': continue
        scores = parse_scores_file(scores_file_path)
        x = [(float(scores[t]['ndcg']) if t in scores else 0) for t in all_topics]
        print(name, x)
        data[name] = x

    fig, ax = plt.subplots(1, 1, figsize=(25,15))
    ax.tick_params(axis='x', rotation=45)
    print(data)
    bar_plot(ax, data)
    plt.xticks(range(len(all_topics)), all_topics)
    save_path = './visualization/compare.png'
    print(f'Saving figure to {save_path} ...')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = './visualization/compare.eps'
    print(f'Saving figure to {save_path} ...')
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'visualize_file': visualize_file,
        'visualize_score_files_in_bar_graph': visualize_score_files_in_bar_graph
    })
