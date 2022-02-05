import os
from subprocess import check_output
import argparse


def calculated_ndcg(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        cmd = [trec_eval_tool, qre_file_path, res_directory+file, "-m", "ndcg"]
        print(' '.join(cmd))
        output = check_output(cmd)
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_map(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        cmd = [trec_eval_tool, qre_file_path, res_directory+file, "-l2", "-m", "map"]
        print(' '.join(cmd))
        output = check_output(cmd)
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_p_at_10(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        cmd = [trec_eval_tool, qre_file_path, res_directory + file, "-l2", "-m", "P"]
        print(' '.join(cmd))
        output = check_output(cmd)
        output = output.decode('utf-8').split("\n")[1]
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_bpref(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        cmd = [trec_eval_tool, qre_file_path, res_directory + file, "-l2", "-m", "bpref"]
        print(' '.join(cmd))
        output = check_output(cmd)
        output = output.decode('utf-8').split("\n")[0]
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file):
    file_res = open(evaluation_result_file, "w")
    res_ndcg = calculated_ndcg(prim_result_dir, trec_eval_tool, qre_file_path)
    res_map = calculated_map(prim_result_dir, trec_eval_tool, qre_file_path)
    res_p10 = calculated_p_at_10(prim_result_dir, trec_eval_tool, qre_file_path)
    res_bpref = calculated_bpref(prim_result_dir, trec_eval_tool, qre_file_path)
    file_res.write("System\tnDCG'\tmAP'\tp@10\tBPref\n")
    for sub in res_ndcg:
        file_res.write(str(sub)+"\t"+str(res_ndcg[sub])+"\t"+str(res_map[sub])+"\t"+str(res_p10[sub])+"\t"+str(res_bpref[sub])+"\n")
    file_res.close()


def main():
    """
    Sample command :
    python3 task1_get_results.py -eva "trec_eval" -qre "qrel_task1.tsv" -pri "/All_Trec_Prime/" -res "task1_result.tsv"
    """
    parser = argparse.ArgumentParser(description='Specify the trec_eval file path, qrel file, '
                                                 'prime results directory and result file  path')

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
