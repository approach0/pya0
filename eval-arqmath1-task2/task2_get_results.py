import os
from subprocess import check_output, CalledProcessError
import argparse


def calculated_ndcg(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory+file, "-m", "ndcg"])
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_map(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory+file, "-l2", "-m", "map"])
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_p_at_10(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory + file, "-l2", "-m", "P"])
        output = output.decode('utf-8').split("\n")[1]
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_bpref(res_directory, trec_eval_tool, qre_file_path, nojudge):
    result = {}
    for file in os.listdir(res_directory):
        output = check_output([trec_eval_tool, qre_file_path, res_directory + file, "-l2", "-m", "bpref"])
        output = output.decode('utf-8')
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = [score, '-']
        if not nojudge:
            cmd = ['python', '-m', 'pya0.judge_rate', qre_file_path, res_directory + file, '--fixfrac', '1000']
            print(' '.join(cmd))
            judge_rate = '-'
            try:
                judge_rate = check_output(cmd).decode('utf-8').strip()
            except CalledProcessError:
                print('WARNING', 'pya0.judge_rate cannot be invoked successfully!')
            result[submission][1] = judge_rate
    return result


def get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file, nojudge):
    file_res = open(evaluation_result_file, "w")
    res_ndcg = calculated_ndcg(prim_result_dir, trec_eval_tool, qre_file_path)
    res_map = calculated_map(prim_result_dir, trec_eval_tool, qre_file_path)
    res_p10 = calculated_p_at_10(prim_result_dir, trec_eval_tool, qre_file_path)
    res_bpref = calculated_bpref(prim_result_dir, trec_eval_tool, qre_file_path, nojudge)
    file_res.write("Run nDCG' mAP' p@10 bpref judge_rate\n")
    for sub in res_ndcg:
        file_res.write(str(sub)+" "+str(res_ndcg[sub])+" "+str(res_map[sub])+" "+str(res_p10[sub])+" "+" ".join(res_bpref[sub])+"\n")
    file_res.close()


def main():
    """
    Sample command :
     python task2_get_results.py -eva trec_eval -qre qrel_task2_2021.tsv -pri Task2_2021_Prime/ -res 2021_task2.tsv
    """
    parser = argparse.ArgumentParser(description='Specify the trec_eval file path, qrel file, '
                                                 'deduplicate results directory and result file path')

    parser.add_argument('-eva', help='trec_eval tool file path', required=True)
    parser.add_argument('-qre', help='qrel file path', required=True)
    parser.add_argument('-pri', help='prime results directory', required=True)
    parser.add_argument('-res', help='evaluation result file', required=True)
    parser.add_argument('-nojudge', help='no judge rate calc', default=False, required=False, action='store_true')

    args = vars(parser.parse_args())
    trec_eval_tool = args['eva']
    qre_file_path = args['qre']
    prim_result_dir = args['pri']+"/"
    evaluation_result_file = args['res']

    get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file, nojudge=args['nojudge'])


if __name__ == "__main__":
    main()
