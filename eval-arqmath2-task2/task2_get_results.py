import os
from subprocess import check_output
import argparse


def calculated_ndcg(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory+file, "-m", "ndcg"])
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prim_")[1]
        result[submission] = score
    return result


def calculated_map(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory+file, "-l2", "-m", "map"])
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prim_")[1]
        result[submission] = score
    return result


def calculated_p_at_10(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory + file, "-l2", "-m", "P"])
        output = output.decode('utf-8').split("\n")[1]
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prim_")[1]
        result[submission] = score
    return result


def get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file):
    file_res = open(evaluation_result_file, "w")
    res_ndcg = calculated_ndcg(prim_result_dir, trec_eval_tool, qre_file_path)
    res_map = calculated_map(prim_result_dir, trec_eval_tool, qre_file_path)
    res_p10 = calculated_p_at_10(prim_result_dir, trec_eval_tool, qre_file_path)
    file_res.write("System\tnDCG'\tmAP'\tp@10\n")
    for sub in res_ndcg:
        file_res.write(str(sub)+"\t"+str(res_ndcg[sub])+"\t"+str(res_map[sub])+"\t"+str(res_p10[sub])+"\n")
    file_res.close()


def main():
    """
    Sample command :
     python task2_get_results.py -eva trec_eval -qre qrel_task2_2021.tsv -pri Task2_2021_Prime/ -res 2021_task2.tsv
    """
    parser = argparse.ArgumentParser(description='Specify the trec_eval file path, qrel file, '
                                                 'deduplicate results directory and result file file path')

    parser.add_argument('-eva', help='trec_eval tool file path', required=True)
    parser.add_argument('-qre', help='qrel file path', required=True)
    parser.add_argument('-pri', help='prime results directory', required=True)
    parser.add_argument('-res', help='evaluation result file', required=True)
    args = vars(parser.parse_args())
    trec_eval_tool = args['eva']
    qre_file_path = args['qre']
    prim_result_dir = args['pri']
    evaluation_result_file = args['res']

    get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file)


if __name__ == "__main__":
    main()
