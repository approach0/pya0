import argparse
import logging
import csv


LOGGER = logging.getLogger(__name__)


def read_task3_result_file(file_path, max_answer_length=1200):
    """
    Reading input file in ARQMath format for ARQMath Task 3
    @param file_path: file path to input file
    @return: dict of topic ids and results
    """
    result = dict()
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for line_number, row in enumerate(csv_reader):
            line_number += 1
            topic_id, rank, score, run_id, sources, answer = row
            if answer.count('$') % 2 == 1:
                LOGGER.warning(f'An odd number of dollar signs ($) in answer to topic {topic_id} '
                               f'on line {line_number}. This may indicate invalid TeX code.')
            if len(answer) > max_answer_length:
                raise ValueError(f'Answer to topic {topic_id} on line {line_number} contains '
                                 f'{len(answer)} Unicode characters, but at most '
                                 f'{max_answer_length} were expected.')
            rank, score = int(rank), float(score)
            if rank != 1:
                LOGGER.warning(f'Answer to topic {topic_id} on line {line_number} has rank '
                               f'{rank}. Ranks of all answers should be 1.')
            if topic_id in result:
                *_, previous_line_number = result[topic_id]
                raise ValueError(f'Topic {topic_id} has at least two different answers, the first '
                                 f'on line {previous_line_number} and the second on line '
                                 f'{line_number}. All topics should only have one answer.')
            result[topic_id] = (topic_id, rank, score, run_id, sources, answer, line_number)
    return result


def main():
    """
    example: python3 validate_task3_results.py -in "TangentS_task3_2021.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Convert results from ARQMath Task 3')
    parser.add_argument('-in', help='Input result file in ARQMath format for ARQMath Task 3', required=True)
    args = vars(parser.parse_args())
    input_file = args['in']

    _ = read_task3_result_file(input_file)
    LOGGER.info(f'{input_file} validates.')


if __name__ == "__main__":
    main()
