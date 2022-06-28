from pathlib import Path
from statistics import mean
import argparse
import csv
import logging


LOGGER = logging.getLogger(__name__)


def read_task3_result_file(file_path):
    """
    Reading input results file in ARQMath format for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, *_ = row
            yield topic_id


def read_task3_map_file(file_path, expected_run_name):
    """
    Reading map file from topic IDs and run names to document IDs for ARQMath Task 3
    @param file_path: file path to input file
    @param expected_run_name: run name of the currently evaluated results
    @return: iterable of topic ids and answer ids
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            run_name, topic_id, answer_id = row
            if run_name == expected_run_name:
                yield (topic_id, answer_id)


def read_task3_qrel_file(file_path):
    """
    Reading input file with relevance judgements for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, and relevance judgements
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, _, answer_id, relevance_judgement = row
            relevance_judgement = int(relevance_judgement)
            yield ((topic_id, answer_id), relevance_judgement)


def main():
    """
    example: python3 evaluate_task3_results.py -in "Baseline2022-task3-GPT3-auto-both-generate-P.tsv"
                                               -map "teams_answer_id.tsv"
                                               -qrel "qrel_task3_2022_official_complete.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Compute Task 3 manual evaluation measures (AR, P@1) for Task 3 results')
    parser.add_argument('-in',
                        help='Input result file in ARQMath format for ARQMath Task 3',
                        required=True)
    parser.add_argument('-map',
                        help='Input map file from topic IDs and run names to synthetic answer IDs for ARQMath Task 3',
                        required=True)
    parser.add_argument('-qrel',
                        help=('Input file with complete relevance judgements (including 5: system '
                              'failure and 6: do not know judgements) for ARQMath Task 3'),
                        required=True)

    args = vars(parser.parse_args())
    result_file = args['in']
    run_name = Path(result_file).stem
    map_file = args['map']
    qrel_file = args['qrel']

    map_dict = dict(read_task3_map_file(map_file, run_name))
    qrel_dict = dict(read_task3_qrel_file(qrel_file))

    missing_topics = set()
    result_dict = dict()
    for topic_id in read_task3_result_file(result_file):
        try:
            answer_id = map_dict[topic_id]
            if topic_id in result_dict:
                raise ValueError(f'Repeated topic {topic_id} in {result_file}')
            result_dict[topic_id] = answer_id
        except KeyError:
            missing_topics.add(topic_id)

    missing_topics = set()
    judgements = []
    for topic_id, answer_id in sorted(result_dict.items()):
        try:
            judgement = qrel_dict[topic_id, answer_id]
            if judgement < 4:  # relevance judgement
                judgements.append(judgement)
            elif judgement == 5:  # system failure
                judgements.append(0)
            elif judgement == 6:  # i don't know
                judgements.append(0)
            else:
                raise ValueError(f'Unknown judgement value {judgement}, expected 0-3, 5, or 6')
        except KeyError:
            missing_topics.add(topic_id)

    #if missing_topics:
    #    LOGGER.warning(f'Results for {len(missing_topics)} topics had no judgements: {sorted(missing_topics)}')
    #    LOGGER.warning(f'Running the evaluation using just {len(judgements)} topics')

    average_relevance = mean(float(judgement) for judgement in judgements)
    precision_at_one = mean(1.0 if judgement > 1 else 0.0 for judgement in judgements)

    print(f'AR:  {average_relevance:.3f}')
    print(f'P@1: {precision_at_one:.3f}')


if __name__ == "__main__":
    main()
