"""
This python script is for converting the runs for task 2 of ARQMath 2020 to retrieval result format that is compatible
to qrel file and can be used for evaluation.

@author: Behrooz Mansouri
@email: bm3302@rit.edu
"""
import operator
import sys
import os
import csv
import argparse
# from math_tan.math_extractor import MathExtractor
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# csv.field_size_limit(sys.maxsize)

def read_latex_files(latex_dir):
    """
    Reading the latex representation of formulas in a dictionary of formula id and latex
    @param latex_dir: the directory in which the latex representations provided by the organizers are located
    @return: dictionary (formula id, latex)
    """
    dic_formula_latex = {}
    for filename in os.listdir(latex_dir):
        with open(latex_dir + filename, mode='r', newline='', encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t', quotechar='"')
            next(csv_reader)
            for row in csv_reader:
                formula_id = int(row[0])
                latex = row[5]
                "removes any white space from the formula latex string"
                latex = "".join(latex.split())
                "the formulas in the comments are ignored"
                if row[3] == "comment":
                    continue
                dic_formula_latex[formula_id] = latex
    return dic_formula_latex


def read_slt_file(slt_file_path):
    """
    Reading the formula slt representations in dictionary, returns dictionary (formula id, slt string)
    @param slt_file_path: file path of the slt string provided by the organizers
    @return: dictionary of (formula id, slt string)
    """
    dic_formula_slt = {}
    with open(slt_file_path, mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            formula_id = int(row[0])
            slt_string = row[2]
            dic_formula_slt[formula_id] = slt_string
    return dic_formula_slt


def read_visual_qrel(qrel_file_path):
    """
    Reading the visual qrel file into a dictionary to be eliminate the unvisited formulas
    @param qrel_file_path: visual qrel file
    @return: dictionary of qrel file
    """
    res_map = {}
    result_file = open(qrel_file_path, newline='', encoding="utf-8")
    csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        topic_id = row[0]
        visual_id = row[2]
        if topic_id in res_map:
            res_map[topic_id].append(visual_id)
        else:
            res_map[topic_id] = [visual_id]
    return res_map


def order_by_score(file_name):
    """
    Takes in the retrieval result file, reads the files and return the retrieval results sorted by scores.
    @param file_name: retrieval results file
    @return: dictionary of (topic id, dictionary (formula id, score)) and the run id.
    """
    res = {}
    result_file = open(file_name, newline='', encoding="utf-8")
    print(file_name)
    csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        topic_id = row[0]
        formula_id = int(row[1])
        if len(row) == 5:
            score = float(row[3])
            run_id = row[4]
        else:
            score = float(row[4])
            run_id = row[5]
        if topic_id in res:
            res[topic_id][formula_id] = score
        else:
            res[topic_id] = {formula_id: score}

    for topic_id in res:
        sorted_dict = dict(sorted(res[topic_id].items(), key=operator.itemgetter(1), reverse=True))
        res[topic_id] = sorted_dict

    return res, run_id


def read_visual_file(visual_file):
    """
    Reading the visual @param visual_file: visual file path @return: two dictionaries in forms of "formula id:slt
    string" and "formula id:latex string"(with white spaces removed)
    """
    slt_dict = {}
    latex_dict = {}
    result_file = open(visual_file, newline='', encoding="utf-8")
    csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        visual_id = row[0]
        slt_string = row[1]
        latex_string = row[2]
        if slt_string != "None":
            slt_dict[slt_string] = visual_id
        else:
            latex_dict[latex_string] = visual_id
    return slt_dict, latex_dict


def get_deduplicate(run_file, qrel_topic_id_lst_visual_ids, slt_string_visual_dict,
                    latex_string_visual_dict, dic_formula_id_latex, dic_formula_id_slt):
    ordered_results_by_score, run_id = order_by_score(run_file)
    final_result = {}
    for topic_id in ordered_results_by_score:
        "topic not in the assessed ones"
        if topic_id not in qrel_topic_id_lst_visual_ids:
            continue
        visited_visual_id = []
        current_topic_results = {}
        for formula_id in ordered_results_by_score[topic_id]:
            # Checking if the visual id is available using the slt string
            if formula_id in dic_formula_id_slt and dic_formula_id_slt[formula_id] in slt_string_visual_dict:
                visual_id = slt_string_visual_dict[dic_formula_id_slt[formula_id]]
            # Checking if the visual id is available using the latex string
            elif formula_id in dic_formula_id_latex and dic_formula_id_latex[formula_id] in latex_string_visual_dict:
                visual_id = latex_string_visual_dict[dic_formula_id_latex[formula_id]]
            else:
                # print(formula_id)
                continue
            "Check if it has been assessed"
            if visual_id not in qrel_topic_id_lst_visual_ids[topic_id]:
                continue
            "In the deduplicated we only considered the first one seen and as the lists are sorted by score we just" \
            "have to iterate to make sure that will happen."
            if visual_id in visited_visual_id:
                continue
            visited_visual_id.append(visual_id)
            score = ordered_results_by_score[topic_id][formula_id]
            current_topic_results[visual_id] = score
        final_result[topic_id] = current_topic_results
    return final_result, run_id


def write_deduplicated(file_path, deduplicated_dict, run_id):
    result_file = open(file_path, "w", newline='', encoding="utf-8")
    csv_writer = csv.writer(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for topic_id in deduplicated_dict:
        temp_dic = deduplicated_dict[topic_id]
        sorted_dict = dict(sorted(temp_dic.items(), key=operator.itemgetter(1), reverse=True))
        rank = 1
        for visual_id in sorted_dict:
            csv_writer.writerow(
                [str(topic_id), "Q0", str(visual_id), str(rank), str(sorted_dict[visual_id]), str(run_id)])
            rank += 1

    result_file.close()


def main():
    """
       Sample command :
       python arqmath_2020_task2_convert_runs.py
       -ru "/runs/"
       -re "/results/"
       -v "visual_id_file.tsv"
       -q "qrel_task2_2020_visual_id.tsv"
       -ld "/latex/"
       -s "formulas_slt_string.tsv"
       """
    parser = argparse.ArgumentParser(description='Specify the directory of run files, qrel file, '
                                                 'deduplicate results directory and result file file path')

    parser.add_argument('-ru', help='directory where the result files are located (Original ARQMath run format)',
                        required=True)
    parser.add_argument('-re', help='directory to save the results', required=True)
    parser.add_argument('-v', help='visual file path', required=True)
    parser.add_argument('-q', help='visual qrel file path', required=True)
    parser.add_argument('-ld', help='directory where the latex tsv files are located', required=True)
    parser.add_argument('-s', help='the file path for slt string', required=True)
    args = vars(parser.parse_args())

    "directory where the result files are located (Original ARQMath run format)"
    run_direcotry = args['ru']
    "directory to save the results"
    result_directory = args['re']

    "visual file path"
    visual_file = args['v']
    "visual qrel file path"
    qrel_file_path = args['q']
    "directory where the latex tsv files are located"
    latex_dir = args['ld']
    "the file path for slt string"
    slt_string_file = args['s']

    qrel_topic_id_lst_visual_ids = read_visual_qrel(qrel_file_path)
    slt_string_visual_dict, latex_string_visual_dict = read_visual_file(visual_file)
    dic_formula_id_latex = read_latex_files(latex_dir)
    dic_formula_id_slt = read_slt_file(slt_string_file)

    for filename in os.listdir(run_direcotry):
        run_file = run_direcotry + filename
        print(run_file)
        final_file = result_directory + "V_" + filename
        final_result, run_id = get_deduplicate(run_file, qrel_topic_id_lst_visual_ids, slt_string_visual_dict,
                                               latex_string_visual_dict, dic_formula_id_latex, dic_formula_id_slt)
        write_deduplicated(final_file, final_result, run_id)


if __name__ == "__main__":
    main()
