import pya0
import copy
import os
import re
from eval import gen_topics_queries
from eval import get_qrels_filepath, parse_qrel_file
from mergerun import parse_trec_file
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


def output_html_topic_run(run_name, qid, query, hits, qrels=None, judged_only=False, scores=None):
    # lookup relevance scores
    for hit in hits:
        docID = hit["trec_docid"]
        qrel_id = f'{qid}/{docID}'
        relev = -1
        if qrels and qrel_id in qrels:
            relev = int(float(qrels[qrel_id]))
        hit['relev'] = relev
    # generate judged-only results, if specified
    if judged_only:
        hits = [h for h in hits if h['relev'] > -1]
    # fix ranks based on score (just like trec_eval does)
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    # prepare output
    RESULTS_PER_PAGE = 100
    tot_pages = len(hits) // RESULTS_PER_PAGE + (len(hits) % RESULTS_PER_PAGE > 0)
    parent_dir = f'./visualization/{run_name}' + ('__judged_only' if judged_only else '')
    scores_str = scores[qid].__str__() if scores and qid in scores else '{ No score available }'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    for page in range(tot_pages):
        start = page * RESULTS_PER_PAGE
        page_hits = hits[start : start + RESULTS_PER_PAGE]
        # start output page
        with open(f'{parent_dir}/{qid}__p{page + 1:03}.html', 'w') as fh:
            mathjax_cdn = "https://cdn.jsdelivr.net/npm/mathjax@3.1.2/es5/tex-chtml.js"
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
            if query is None: query = []
            actual_query = preprocess_query(copy.deepcopy(query), expansion=False)
            for q, aq in zip(query, actual_query):
                if q['type'] == 'term':
                    kw_str = f'{q["str"]} &nbsp;&nbsp; (stemmed: {aq["str"]})'
                else:
                    kw_str = f'[imath]{q["str"]}[/imath]'
                fh.write(f'<li>{kw_str}</li>\n')
            fh.write('</ul>\n')
            # hits
            if judged_only:
                fh.write(f'<h3>Judged Hits (page #{page + 1} / {tot_pages})</h3>\n')
            else:
                fh.write(f'<h3>Hits (page #{page + 1} / {tot_pages})</h3>\n')
            fh.write(f'<h5>Scores: {scores_str}</h5>\n')
            output_html_pagination(fh, qid, page, tot_pages)
            fh.write('<ol>\n')
            for hit in page_hits:
                docID = hit["trec_docid"]
                rank = hit['rank']
                score = hit['score']
                relev = hit['relev']
                fh.write('<li>\n')
                fh.write(f'<b id="{rank}">rank <a href="#{rank}">#{rank}</a>, ' +
                         f'doc #{docID}, score: {score}, relevance: {relev}, </b>\n')
                colors = [f'<b style="background: {degree_color(i)}">{i}</b>' for i in range(4)]
                color = degree_color(relev)
                fh.write('<b>relevance levels: ' + ' '.join(colors) + ':</b>')
                fh.write(f'<p style="background: {color};">{hit["content"]}</p>\n')
                fh.write('</li>\n')
            fh.write(f'</ol>\n')
            output_html_pagination(fh, qid, page, tot_pages)
            # mathJax
            fh.write('<script> window.MathJax ={' +
                "loader: { source: {'[tex]/AMScd': '[tex]/amscd'} }," +
                'tex: { inlineMath: [["[imath]", "[/imath]"]] }' +
            '}; </script>')
            fh.write(f'<script type="text/javascript" src="{mathjax_cdn}"></script>')
            # end document
            fh.write('</body></html>\n')


def visualize_hits(index, run_name, qid, query, hits, qrels=None, scores=None):
    # lookup document content
    for hit in hits:
        docid = hit['docid'] # must be internal docid
        doc = collection_driver.docid_to_doc(index, docid)
        hit['content'] = doc['content']
    # output HTML preview
    if qrels:
        output_html_topic_run(run_name, qid, query, hits, qrels=qrels, judged_only=True, scores=scores)
    if True:
        output_html_topic_run(run_name, qid, query, hits, qrels=qrels, judged_only=False, scores=scores)


def visualize_collection_runs(index, collection, tsv_file_path):
    run_per_topic, _ = parse_trec_file(tsv_file_path)
    scores_file_path = '.'.join(tsv_file_path.split('.')[0:-1]) + '.scores'
    scores = parse_scores_file(scores_file_path)
    run_name = os.path.basename(tsv_file_path)
    qrels_path = get_qrels_filepath(collection)
    print('QRELS:', qrels_path)
    qrels = parse_qrel_file(qrels_path)
    for i, (qid, query, _) in enumerate(gen_topics_queries(collection)):
        print(qid, query)
        topic_hits = run_per_topic[qid] if qid in run_per_topic else []
        collection_driver.TREC_reverse(collection, index, topic_hits)
        visualize_hits(index, run_name, qid, query, topic_hits, qrels=qrels, scores=scores)


def visualize(index, tsv_file_path, collection=None, adhoc_query=None):
    print(f'\n\t Visualize runfile: {tsv_file_path} ...\n')
    if collection and not adhoc_query:
        visualize_collection_runs(index, collection, tsv_file_path)
    elif collection and adhoc_query:
        run_name = os.path.basename(tsv_file_path)
        run_per_topic, _ = parse_trec_file(tsv_file_path)
        for qid, hits in run_per_topic.items():
            collection_driver.TREC_reverse(collection, index, hits)
            visualize_hits(index, run_name, qid, adhoc_query, hits)
    else:
        print('Error: Please specify --collection for visualization.')
        quit(1)


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


def visualize_compare_scores(score_files):
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
