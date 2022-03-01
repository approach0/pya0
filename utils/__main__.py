import os
import sys
import json
import copy
import argparse
import auto_eval
import time
import pickle
import pya0
from .mindex_info import list_indexes
from .eval import run_topics, evaluate_run, evaluate_log
from .msearch import cascade_run, msearch
from .mergerun import concatenate_run_files, merge_run_files
from .l2r import L2R_gen_train_data, L2R_train
import preprocess


def abort_on_non_a0_index(index):
    if isinstance(index, str):
        print('Abort on network index')
        exit(1)


def abort_on_empty_collection(collection):
    if collection is None:
        print('Please specify collection name using --collection')
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utilities for running Approach0 and evaluateion')

    parser.add_argument('--query', type=str, required=False, nargs='+',
        help="Mixed type of keywords, math keywords are written in TeX and wrapped up in dollars")
    parser.add_argument('--stemmer', type=str, required=False,
        help="Specified stemmer name. E.g., lancaster")
    parser.add_argument('--direct-search', type=str, required=False,
        help="Issue direct search query in pickle, and output JSON results")
    parser.add_argument('--select-topic', type=str, required=False,
        help="Select specific topic to run for evaluation")
    parser.add_argument('--filter', type=str, required=False,
        help="Add topic filter layer for training/testing")
    parser.add_argument('--docid', type=int, required=False,
        help="Lookup a raw document from index")
    parser.add_argument('--index', type=str, required=False,
        help="Open index at specified path, a prebuilt index name, or a searchd Web API" +
             " (e.g., http://localhost:8921/search)")
    parser.add_argument('--topk', type=int, required=False,
        help="Keep at most top-K hits in results")
    parser.add_argument('--trec-output', type=str, required=False,
        help="Output TREC-format results")
    parser.add_argument('--verbose', required=False, action='store_true',
        help="Verbose output (showing query structures and merge times)")
    parser.add_argument('--use-fallback-parser', required=False, action='store_true',
        help="Use fallback LaTeXML parser for parsing TeX")
    parser.add_argument('--print-index-stats', required=False,
        action='store_true', help="Print index statistics and abort")
    parser.add_argument('--list-prebuilt-indexes', required=False,
        action='store_true', help="List available prebuilt math indexes and abort")
    parser.add_argument('--collection', type=str, required=False,
        help="Specified collection name so this program can associate its qrels/topics")
    parser.add_argument('--eval-args', type=str, required=False,
        help="Passing extra command line arguments to trec_eval. E.g., '-q -m map -m P.30'")
    parser.add_argument('--visualize-run', type=str, required=False,
        help="Visualize ARQMath Task1 run files")
    parser.add_argument('--visualize-compare-scores', type=str, required=False,
        help="Visualize scores comparison")
    parser.add_argument('--concate-runs', type=str, required=False,
        help="Concatenate run files, format: A,B,n where n is the top hit number that are kept in A")
    parser.add_argument('--merge-runs', type=str, required=False,
        help="Merge and run files (interpolating by scores), format: A,B,alpha where alpha is the weight applied on A")
    parser.add_argument('--read-file', type=str, required=False,
        help="Instead of returning results from search engine, read results from file. E.g., --read-file TREC:tmp.run")
    parser.add_argument('--rm3', type=str, required=False,
        help="Apply RM3 (query expansion using Relevance Model): fbTerms,fbDocs. E.g., '--rm3 20,10'")
    parser.add_argument('--training-data-from-run', type=str, required=False,
        help="Output learning2rank training data")
    parser.add_argument('--learning2rank-train', type=str, required=False,
        help="train learning-to-rank model. E.g., '--learnig2rank-train lambdaMART,90,5,<path>'")
    parser.add_argument('--learning2rank-rerank', type=str, required=False,
        help="apply learning-to-rank model. E.g., '--learnig2rank-rerank lambdaMART,<path>'")
    parser.add_argument('--math-expansion', required=False, action='store_true',
        help="do text expansion for math query keyword(s)")
    parser.add_argument('--kfold', type=int, required=False,
        help="Sample input topics with k-fold validation.")
    parser.add_argument('--auto-eval', type=str, required=False,
        help="Automatically evaluate multiple experiments specified by TSV file. E.g., '--auto-eval <name>'")
    parser.add_argument('--auto-eval-summary', type=str, required=False,
        help="Print automatic evaluation summary in TSV. E.g., '--auto-eval <name>'")

    args = parser.parse_args()

    # overwrite default arguments for running searcher
    verbose = args.verbose if args.verbose else False
    topk = args.topk if args.topk else 1000
    trec_output = args.trec_output if args.trec_output else '/dev/null'

    # enable fallback parser?
    if args.use_fallback_parser:
        print('use fallback parser.')
        pya0.use_fallback_parser(True)

    # use stemmer?
    if args.stemmer:
        print('use stemmer', args.stemmer)
        preprocess.use_stemmer(name=args.stemmer)

    # initial filter layer
    if args.filter:
        cascades = [('filter', [args.filter])]
    else:
        cascades = []

    # initial retrieve layer
    if args.read_file:
        file_format, file_path = args.read_file.split(':')
        cascades.append(('reader', [file_format, file_path]))
    else:
        cascades.append(('first-stage', {
            'first-stage-args': None
        }))

    # add cascade layers
    if args.rm3:
        fbTerms, fbDocs = args.rm3.split(',')
        fbTerms, fbDocs = int(fbTerms), int(fbDocs)
        cascades.append(('rm3', [fbTerms, fbDocs]))

    elif args.learning2rank_rerank:
        fields = args.learning2rank_rerank.split(',')
        method, params = fields[0], fields[1:] if len(fields) > 1 else []
        cascades.append(('l2r', [method, params]))

    # list prebuilt indexes?
    if args.list_prebuilt_indexes:
        list_indexes()
        exit(0)

    # print auto-eval summary?
    elif args.auto_eval_summary:
        abort_on_empty_collection(args.collection)
        name = args.auto_eval_summary
        header, rows = auto_eval.tsv_eval_read('product.tsv')
        def wrap_summary(idx, run_name, _):
            global header
            run_path=f'tmp/{run_name}.run'
            log_path=f'tmp/{run_name}.log'
            if os.path.exists(f'tmp/{run_name}.done'):
                run_header, run_row = evaluate_run(args.collection, run_path)
                log_header, log_row = evaluate_log(args.collection, log_path)
                if idx == 0:
                    header = header + run_header + log_header
                rows[idx] += run_row + log_row
            else:
                print('skip this row')
        auto_eval.tsv_eval_do(header, rows, wrap_summary, prefix=name+'-')
        with open(f'summary-{name}.tsv', 'w') as fh:
            print('#' + '\t'.join(header), file=fh)
            for row in rows:
                print('\t'.join(row), file=fh)
        exit(0)

    # visualize score comparison?
    elif args.visualize_compare_scores:
        from .visualize import visualize_compare_scores
        files = args.visualize_compare_scores.split(':')
        visualize_compare_scores(files)
        exit(0)

    # concatenate run files?
    elif args.concate_runs:
        A, B, n = args.concate_runs.split(',')
        concatenate_run_files(A, B, int(n), topk, verbose=verbose)
        exit(0)

    # merge run files?
    elif args.merge_runs:
        A, B, alpha = args.merge_runs.split(',')
        merge_run_files(A, B, float(alpha), topk, verbose=verbose)
        exit(0)

    # learning to rank?
    elif args.learning2rank_train:
        fields = args.learning2rank_train.split(',')
        method, params = fields[0], fields[1:] if len(fields) > 1 else []
        L2R_train(method, params, output_file=args.trec_output)
        exit(0)

    # open index from specified index path or prebuilt index
    if args.index is None:
        print('No index specified, abort.')
        exit(1)

    elif isinstance(args.index, str) and ':' in args.index:
        import collection_driver
        index = collection_driver.open_index(args.index)

    elif not os.path.exists(args.index):
        index_path = pya0.from_prebuilt_index(args.index, verbose=verbose)
        if index_path is None: # if index name is not registered
            exit(1)
        index = pya0.index_open(index_path, option="r")
    else:
        index_path = args.index
        index = pya0.index_open(index_path, option="r")

    if index is None:
        print(f'index open failed: {index_path}')
        exit(1)

    # direct search
    if args.direct_search:
        with open(args.direct_search, 'rb') as fh:
            query, topk, log = pickle.load(fh)
            print(query, topk, log, file=sys.stderr)
            sys.stderr.flush()
            res = msearch(index, query, topk=topk, log=log)
            print(json.dumps(res, indent=4))
        pya0.index_close(index)
        exit(0)

    # output HTML file
    elif args.visualize_run and not args.query:
        from .visualize import visualize
        visualize(index, args.visualize_run, collection=args.collection)
        exit(0)

    # generate l2r training data
    elif args.training_data_from_run:
        abort_on_non_a0_index(index)
        abort_on_empty_collection(args.collection)
        L2R_gen_train_data(args.collection, index, args.training_data_from_run)
        exit(0)

    # print index stats
    elif args.print_index_stats:
        abort_on_non_a0_index(index)
        print(f' --- index stats ({args.index}) ---')
        pya0.index_print_summary(index)
        exit(0)

    # auto evaluation?
    elif args.auto_eval:
        abort_on_empty_collection(args.collection)
        name = args.auto_eval
        print('reading auto_eval.tsv ...')
        out_tsv_content = auto_eval.tsv_product('auto_eval.tsv')
        out_tsv = 'product.tsv'
        print(f'generating {out_tsv} ...')
        with open(out_tsv, 'w') as fh:
            fh.write(out_tsv_content + '\n')
        print('starting evaluation ...')
        header, rows = auto_eval.tsv_eval_read(out_tsv)
        def wrap_eval(idx, run_name, replaces):
            if os.path.exists(f'tmp/{run_name}.done'):
                print('skip this row')
                return
            auto_eval.replace_source_code('./template', replaces)
            print('Rebuild project in 3 seconds...')
            time.sleep(3)
            auto_eval.remake('..')
            run_topics(index, args.collection,
                output=f'tmp/{run_name}.run',
                log=f'tmp/{run_name}.log',
                topk=topk,
                verbose=False,
                cascades=cascades,
                math_expansion=args.math_expansion,
                fork_search=args.index,
                select_topic=args.select_topic
            )
            with open(f'tmp/{run_name}.done', 'w') as fh:
                pass
        auto_eval.tsv_eval_do(header, rows, wrap_eval, prefix=name+'-')
        exit(0)

    # actually run query
    if args.query:
        # parser query
        query = []
        for kw in args.query:
            kw_type = 'term'
            if kw.startswith('$'):
                kw = kw.strip('$')
                kw_type = 'tex'
            query.append({
                'str': kw,
                'type': kw_type,
                'field': 'content',
            })

        # process initial query
        origin_query = copy.deepcopy(query)
        query = preprocess.preprocess_query(query, expansion=args.math_expansion)
        collection = args.collection if args.collection else 'test'

        if verbose:
            print('[origin query] ', origin_query)
            print('[processed query] ', query)

        # actually run query
        topic_query = ('TEST.0', query, '') # no tags
        hits = cascade_run(index, cascades, topic_query, verbose=verbose,
                           topk=topk, collection=collection, docid=args.docid)
        # print hits
        for hit in hits:
            snippet = hit['field_content']
            del hit['field_content']
            snippet = snippet.replace('<em class="hl">', "\033[91m")
            snippet = snippet.replace('</em>', "\033[0m")
            print("\033[94m" + json.dumps(hit) + "\033[0m")
            print(snippet)
            print()

        # output run file
        if args.trec_output is not None:
            import collection_driver
            from .eval import TREC_output
            collection_driver.TREC_preprocess(collection, index, hits)
            TREC_output(hits, 'TEST.0', append=False, output_file=trec_output)

        # output HTML file
        if args.visualize_run:
            from .visualize import visualize
            visualize(index, args.visualize_run,
                adhoc_query=origin_query, collection=collection)

    elif args.docid:
        abort_on_non_a0_index(index)
        doc = pya0.index_lookup_doc(index, args.docid)
        print(json.dumps(doc, indent=4))

    elif args.collection:
        abort_on_empty_collection(args.collection)

        if args.trec_output is None:
            print('Error: Must specify a TREC output file to run topics')
            exit(1)

        run_topics(index, args.collection,
            output=trec_output,
            topk=topk,
            verbose=verbose,
            trec_eval_args=args.eval_args,
            cascades=cascades,
            math_expansion=args.math_expansion,
            #fork_search=args.index,
            kfold=args.kfold,
            select_topic=args.select_topic
        )
    else:
        print('No --docid, --query --collection specifed, abort.')
        exit(1)
