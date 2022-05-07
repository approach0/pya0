import sys
import pickle
import argparse
sys.path.insert(0, '.')
from pya0.mergerun import parse_trec_file


def TREC_output(run_dict, run_name, output_file="tmp.run"):
    with open(output_file, 'w') as fh:
        for qid in run_dict:
            print("%s %s %s %u %f %s" % (
                qid,
                run_dict[qid][0]["_"],
                run_dict[qid][0]["docid"],
                1,
                run_dict[qid][0]['score'],
                run_name
            ), file=fh)
            fh.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, help='runfile for task 1', required=True)
    parser.add_argument('--best_dict', type=str, help='the pickle file maps answers to the accepted under the same question', default="aid2bestgt1_aid.pkl")
    parser.add_argument('--output_path', type=str, help='path to save the new run file, same format as task1 output', required=True)
    args = parser.parse_args()

    aid2best = pickle.load(open(args.best_dict, 'rb'))

    output_file = open(args.output_path, 'w')

    run_dict, run_name = parse_trec_file(args.run_path)
    new_run_dict = {}
    for qid in run_dict:
        best_aid = 0
        best_score = 0
        sorted_answers = sorted(run_dict[qid], key=lambda x: float(x["score"]), reverse=True)
        got_answer = False

        for a in sorted_answers:
            docid = a["docid"]
            if docid in aid2best and aid2best[docid]:
                new_id = aid2best[docid]
                new_run_dict[qid] = [{
                    "docid": new_id,
                    "_": "best",
                    "rank": a["rank"],
                    "score": a["score"],
                    }]
                got_answer = True
                break
        if not got_answer:
            new_run_dict[qid] = [{
                "docid":sorted_answers[0]["docid"],
                "_": "origin",
                "rank": sorted_answers[0]["rank"],
                "score": sorted_answers[0]["score"]
            }]

    TREC_output(new_run_dict, run_name, output_file=args.output_path)






