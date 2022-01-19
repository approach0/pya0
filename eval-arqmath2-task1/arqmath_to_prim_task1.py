import csv
import os
import argparse


def convert_result_files_to_trec(submission_dir, qrel_result_dic, trec_dir, prime_dir):
    """
    this method reads all the results files and convert them to trec_format, both with and without unjudged posts
    @param submission_dir: the dirctory where the original submissions are located
    @param qrel_result_dic: the qrel file for task 1 file path
    @param trec_dir: the destination to keep the trec_formatted files
    @param prime_dir: the destination to keep the prime_trec_formatted files
    """
    for file in os.listdir(submission_dir):
        topic_result = {}
        result_file = open(submission_dir + file, newline='', encoding="utf-8")
        csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

        "Trec_formatted"
        result_file1 = open(trec_dir + file, "w", newline='', encoding="utf-8")
        csv_writer1 = csv.writer(result_file1, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        "Prim file"
        result_file2 = open(prime_dir + "prime_" + file, "w", newline='', encoding="utf-8")
        csv_writer2 = csv.writer(result_file2, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

        for row in csv_reader:
            topic_id = row[0]
            post_id = row[1]
            if topic_id in topic_result:
                if post_id in topic_result[topic_id]:
                    "When working with trec if the retrieved results for a topic are not unique, you will get errors"
                    continue
                else:
                    topic_result[topic_id].append(post_id)
            else:
                topic_result[topic_id] = [post_id]
            temp = [row[0], "Q0"]
            temp.extend(row[1:5])
            csv_writer1.writerow(temp)
            topic_id = row[0]
            "--------------------------------------------------------------------"
            "Creating Prim file"
            if topic_id not in qrel_result_dic:
                continue
            if post_id not in qrel_result_dic[topic_id]:
                continue
            else:
                csv_writer2.writerow(temp)
        result_file1.close()
        result_file2.close()


def read_qrel_to_dictionary(qrel_file_path):
    """
    Reading the qrel file into a dictionary of topics and annotated answer_ids
    @param qrel_file_path: qrel file path
    @return: dictionart of topics and list of answer_ids
    """
    res_map = {}
    result_file = open(qrel_file_path, newline='', encoding="utf-8")
    csv_reader = csv.reader(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        topic_id = row[0]
        post_id = row[2]
        if topic_id in res_map:
            res_map[topic_id].append(post_id)
        else:
            res_map[topic_id] = [post_id]
    return res_map


def main():
    """
    Sample command:
    python3 arqmath_to_prim_task1.py -qre qrel_partial_task1 -sub "/ARQMath Task 1/All_results/"  -tre "/ARQMath Task 1/All_Trec/" -pri "/ARQMath Task 1/All_Trec_Prime/"
    """
    parser = argparse.ArgumentParser(description='Takes in the qrel file and the original submitted file to arqmath'
                                                 'and creates the trec_formatted and prime files')

    parser.add_argument('-qre', help='qrel file path', required=True)
    parser.add_argument('-sub', help='Directory path in which there are the submitted files', required=True)
    parser.add_argument('-tre', help='Directory path to save Trec files')
    parser.add_argument('-pri', help='Directory path to save prime files')
    args = vars(parser.parse_args())

    qrel_file_path = args['qre']  
    source_submitted_files_dir = args['sub']  
    destination_trec_formatted_dir = args['tre'] 
    destination_trec_formattd_prime_dir = args['pri'] 

    qrel_dictionary = read_qrel_to_dictionary(qrel_file_path)
    convert_result_files_to_trec(source_submitted_files_dir, qrel_dictionary,
                                 destination_trec_formatted_dir,
                                 destination_trec_formattd_prime_dir)


if __name__ == "__main__":
    main()
