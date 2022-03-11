import os
import fire
from mergerun import parse_trec_file
import numpy as np


def each_run_file(all_run_files):
    for run_file in all_run_files:
        if not os.path.isfile(run_file):
            continue
        yield run_file


def main(*all_run_files, kfold=5):
    all_topic_ids = set()
    print('Reading all topics...')
    for run_file in each_run_file(all_run_files):
        run_per_topic, _ = parse_trec_file(run_file)
        all_topic_ids.update(run_per_topic.keys())
    all_topic_ids = np.array(list(all_topic_ids))
    np.random.shuffle(all_topic_ids)
    folds = np.array_split(all_topic_ids, kfold)
    print(folds)
    for k in range(len(folds)):
        test_fold = set(folds[k])
        for run_file in each_run_file(all_run_files):
            print(f'fold{k}, runfile: {run_file}')
            holdout_file = f'{run_file}.fold{k}holdout'
            test_file = f'{run_file}.fold{k}foldtest'
            with open(run_file, 'r') as fh, open(holdout_file, 'w') as holdout_fh, open(test_file, 'w') as test_fh:
                for line in fh:
                    line = line.rstrip()
                    fields = line.split()
                    topic = fields[0]
                    if topic in test_fold:
                        print(line, file=test_fh)
                    else:
                        print(line, file=holdout_fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
