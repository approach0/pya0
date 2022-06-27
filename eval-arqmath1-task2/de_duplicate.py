import argparse
import csv
import operator
import os
import sys

csv.field_size_limit(sys.maxsize)


def read_qrel(qrel_file_path):
    res = {}
    result_file = open(qrel_file_path, newline='', encoding="utf-8")
    csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        topic_id = row[0]
        visual_id = int(row[2])
        if topic_id in res:
            res[topic_id].append(visual_id)
        else:
            res[topic_id] = [visual_id]
    return res


def read_visual_files(lst_dir, vid_index):
    dic_formula_visual_id = {}
    for file in os.listdir(lst_dir):
        print(lst_dir + "/" + file)
        with open(lst_dir + "/" + file, newline='', encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            next(csv_reader)
            for row in csv_reader:
                formula_id = int(row[0])
                visual_id = int(row[vid_index])
                dic_formula_visual_id[formula_id] = visual_id
    return dic_formula_visual_id


def read_raw_submission_file(submission_file_path):
    with open(submission_file_path, newline='', encoding="utf-8") as result_file:
        csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        result_dic = {}
        for row in csv_reader:
            topic_id = row[0]
            formula_id = int(row[1])
            score = float(row[4])
            if topic_id in result_dic:
                result_dic[topic_id][formula_id] = score
            else:
                result_dic[topic_id] = {formula_id: score}
    return result_dic, row[5]


def deduplicate(dic_formula_visual_ids, submission_file_path, topic_visual_lst):
    deduplicated_result = {}
    result_dict, run = read_raw_submission_file(submission_file_path)
    for topic_id in result_dict:
        if topic_id not in topic_visual_lst:
            continue
        temp_result_dic = result_dict[topic_id]
        sorted_dict = dict(sorted(temp_result_dic.items(), key=operator.itemgetter(1), reverse=True))
        visited_visual_ids = []
        temp_dict = {}
        for formula_id in sorted_dict:
            if formula_id not in dic_formula_visual_ids:
                print(formula_id)
                continue
            visual_id = dic_formula_visual_ids[formula_id]
            if visual_id in visited_visual_ids:
                continue
            score = sorted_dict[formula_id]
            temp_dict[visual_id] = score
            visited_visual_ids.append(visual_id)
        deduplicated_result[topic_id] = temp_dict
    return deduplicated_result, run


def deduplicated_prim_file(submission_dir, result_dir, file, topic_visual_lst, dic_formula_visual_ids):
    """
    Takes in the retrieval result file, reads the files and return the retrieval results sorted by scores.
    @param file_name: retrieval results file
    @return: dictionary of (topic id, dictionary (formula id, score)) and the run id.
    """
    deduplicated_result, run_id = deduplicate(dic_formula_visual_ids, submission_dir + file, topic_visual_lst)
    prim_dic = {}
    for topic_id in topic_visual_lst:
        topic_prim_dic = {}
        lst_visual_ids = topic_visual_lst[topic_id]
        if topic_id not in deduplicated_result:
            continue
        temp_dic = deduplicated_result[topic_id]
        for visual_id in temp_dic:
            if visual_id not in lst_visual_ids:
                continue
            topic_prim_dic[visual_id] = temp_dic[visual_id]
        topic_prim_dic = dict(sorted(topic_prim_dic.items(), key=operator.itemgetter(1), reverse=True))
        prim_dic[topic_id] = topic_prim_dic

    with open(result_dir + "prime_" + file, "w", newline='', encoding="utf-8") as result_file:
        csv_writer = csv.writer(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for topic_id in prim_dic:
            temp_dic = prim_dic[topic_id]
            rank = 1
            for visual_id in temp_dic:
                csv_writer.writerow([str(topic_id), "Q0", str(visual_id), str(rank), str(temp_dic[visual_id]),
                                     str(run_id)])
                rank += 1


def main():
    """
    Sample command:
    python3 de_duplication_2021.py
    -qre "qrel_task2_2021.tsv"
    -tsv "/latex_representation_v2/"
    -sub "/ARQMath 2021 Submission/Z_Task2_2021/"
    -pri "/ARQMath 2021 Submission/Z_Task2_2021_Prime/"
    """
    parser = argparse.ArgumentParser(description='Takes in the qrel file and the original submitted file to arqmath'
                                                 'and creates the prim files')

    parser.add_argument('-qre', help='qrel file path', required=True)
    parser.add_argument('-tsv', help='Directory path in which there are the latex tsv files', required=True)
    parser.add_argument('-sub', help='Directory path in which there are the submitted files', required=True)
    parser.add_argument('-v', help='visual id index; old version(2): 4 / new version(3) 6')
    parser.add_argument('-pri', help='Directory path to save prime files')
    args = vars(parser.parse_args())

    tsv_dir = args['tsv'] + "/"
    submission_dir = args['sub'] + "/"
    result_dir = args['pri'] + "/"
    visual_index = int(args['v'])
    qrel_file_path = args['qre']
    dic_formula_visual_ids = read_visual_files(tsv_dir, visual_index)
    topic_visual_lst = read_qrel(qrel_file_path)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for file in os.listdir(submission_dir):
        deduplicated_prim_file(submission_dir, result_dir, file, topic_visual_lst, dic_formula_visual_ids)


if __name__ == "__main__":
    main()
