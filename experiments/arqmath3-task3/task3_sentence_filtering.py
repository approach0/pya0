import argparse
from nltk.tokenize import sent_tokenize

leng_threshold = 1195

def parse_run(filename):
    f = open(filename, 'r')
    query2sentences = {}
    for line in f:
        qid, rank, score, rid, sources, answer = line.strip().split("\t")
        answer = answer.replace('"', "'")
        answer = '"{}"'.format(answer[1:-1])
        if len(answer) > leng_threshold or len(answer) == 0:
            continue
        if answer.count('$') % 2 == 1:
            continue
        if qid not in query2sentences:
            query2sentences[qid] = [["1", score, rid, sources, answer]]
        else:
            query2sentences[qid].append(["1", score, rid, sources, answer])

    return query2sentences, rid

#def chunk_sent(sent, threshold=leng_threshold):
#    sentences = sent_tokenize(sent)
#    ret = ""
#    for s in sentences:
#        if len(ret + " " + s) < threshold:
#            ret = ret + " " + s
#        else:
#            return ret
#    if len(ret):
#        return ret
#    else:
#        print(sent)
#        return sent[:leng_threshold]

def select_highest_score(results, default_rid="maprun-task3.run"):
    selected_results = {}
    for qid in results:
        #current_best = ("1", "0", default_rid, ("", "0", "0"), "") # rank, score, rid, scource, answer
        current_best = results[qid][0]
        for sentence in results[qid]:
            if float(sentence[1]) > float(current_best[1]) and len(sentence[-1]) < leng_threshold:
                current_best = sentence
        selected_results[qid] = current_best
        selected_results[qid][0] = "1"
#        selected_results[qid][-1] = chunk_sent(selected_results[qid][-1])
    return selected_results

def select_highest_post_longest(results, default_rid="maprun-task3.run"):
    selected_results = {}
    for qid in results:
        sorted_answers = sorted(results[qid], key=lambda x: float(x[1]), reverse=True)
        docid = sorted_answers[0][3][0]
        current_best = sorted_answers[0]
        for a in sorted_answers:
            if a[3] == docid:
                if len(a[-1]) < leng_threshold and len(a[-1]) > len(current_best[-1]):
                    current_best = a
        selected_results[qid] = current_best
        selected_results[qid][0] = "1"
#        selected_results[qid][-1] = chunk_sent(current_best[-1])
    return selected_results

def save_results(results, out_filename):
    f = open(out_filename,'w')
    for qid in results:
        f.write('{}\t{}\t{}\n'.format(qid, "\t".join(results[qid][:-1]), results[qid][-1]))
    f.close() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, help='runfile for task 1', required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--mode', type=str, help='choose from: highest_score, highest_post_longest')
    args = parser.parse_args()

    results, rid = parse_run(args.run_path)
    #if len(results) < 100:
    #    print(len(results))
    #    input()

    if args.mode == "highest_score":
        results = select_highest_score(results, rid)

    elif args.mode == "highest_post_longest":
        results = select_highest_post_longest(results, rid)

    save_results(results, args.output_path)
