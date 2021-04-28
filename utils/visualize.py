import pya0
import os
from .eval import gen_topics_queries
from .eval import get_qrels_filepath, parse_qrel_file
from .mergerun import parse_trec_file


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
    else:
        return 'red'


def output_html_topic_run(run_name, qid, query, hits, qrels=None, judged_only=False, scores=None):
    # lookup relevance scores
    for hit in hits:
        docID = hit["docid"]
        qrel_id = f'{qid}/{docID}'
        relev = -1
        if qrels and qrel_id in qrels:
            relev = int(qrels[qrel_id])
        hit['relev'] = relev
    # generate judged-only results, if specified
    if judged_only:
        hits = [h for h in hits if h['relev'] > -1]
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
            for q in query:
                kw_str = q['str'] if q['type'] == 'term' else f'[imath]{q["str"]}[/imath]'
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
                docID = hit["docid"]
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
            fh.write('<script> window.MathJax ={ tex: { inlineMath: [["[imath]", "[/imath]"]] } }; </script>')
            fh.write(f'<script type="text/javascript" src="{mathjax_cdn}"></script>')
            # end document
            fh.write('</body></html>\n')


def visualize_run(index, collection, tsv_file_path):
    run_per_topic, _ = parse_trec_file(tsv_file_path)
    scores_file_path = '.'.join(tsv_file_path.split('.')[0:-1]) + '.scores'
    scores = parse_scores_file(scores_file_path)
    run_name = os.path.basename(tsv_file_path)
    qrels = parse_qrel_file(get_qrels_filepath(collection))
    for i, (qid, query, _) in enumerate(gen_topics_queries(collection)):
        print(qid, query)
        topic_hits = run_per_topic[qid] if qid in run_per_topic else []
        # lookup document content
        for hit in topic_hits:
            trec_docid = hit['docid']
            doc = pya0.index_lookup_doc(index, trec_docid)
            doc = pya0.index_lookup_doc(index, int(doc['extern_id']))
            hit['content'] = doc['content']
        output_html_topic_run(run_name, qid, query, topic_hits, qrels=qrels, judged_only=False, scores=scores)
        output_html_topic_run(run_name, qid, query, topic_hits, qrels=qrels, judged_only=True, scores=scores)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
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


def visualize_compare_scores(scores_file_path1, scores_file_path2):
    scores1 = parse_scores_file(scores_file_path1)
    scores2 = parse_scores_file(scores_file_path2)
    all_topics = list(set(list(scores1.keys()) + list(scores2.keys())))
    all_topics = sorted(all_topics, key=lambda x: x.__str__())
    import matplotlib.pyplot as plt

    y = [(float(scores1[t]['ndcg']) if t in scores1 else -1) for t in all_topics]
    z = [(float(scores2[t]['ndcg']) if t in scores2 else -1) for t in all_topics]

    data = {
        scores_file_path1: y,
        scores_file_path2: z,
    }

    fig, ax = plt.subplots(1, 1, figsize=(25,15))
    ax.tick_params(axis='x', rotation=45)
    bar_plot(ax, data)
    plt.xticks(range(len(all_topics)), all_topics)
    save_path = './visualization/compare.png'
    print(f'Saving PNG to {save_path} ...')
    plt.savefig(save_path)
    plt.show()
