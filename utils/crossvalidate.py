import os
import re
import fire
import random
import numpy as np
from mergerun import parse_trec_file
from collections import defaultdict


def each_run_file(all_run_files):
    for run_file in all_run_files:
        if not os.path.isfile(run_file):
            continue
        yield run_file


def split_run_files(*all_run_files, kfold=5, seed=123):
    all_topic_ids = dict() # do not use set(), we need order here!
    print('Reading all topics...')
    for run_file in each_run_file(all_run_files):
        run_per_topic, _ = parse_trec_file(run_file)
        all_topic_ids.update(dict(run_per_topic.items()))
    all_topic_ids = np.array(list(all_topic_ids))
    print('Shuffle file using seed:', seed)
    np.random.seed(seed)
    np.random.shuffle(all_topic_ids) # random shuffle!!!
    folds = np.array_split(all_topic_ids, kfold)
    print(folds)
    for k in range(len(folds)):
        test_fold = set(folds[k])
        for run_file in each_run_file(all_run_files):
            print(f'fold{k}, runfile: {run_file}')
            train_file = f'{run_file}.fold{k}train'
            test_file = f'{run_file}.fold{k}test'
            with open(run_file, 'r') as fh, \
                open(train_file, 'w') as train_fh, \
                open(test_file, 'w') as test_fh:
                for line in fh:
                    line = line.rstrip()
                    fields = line.split()
                    topic = fields[0]
                    if topic in test_fold:
                        print(line, file=test_fh)
                    else:
                        print(line, file=train_fh)


def cross_validate_tsv(tsv_file, name_field=0, score_field=1,
    verbose=True, postfix=None):
    """
    Given a tsv file with the following fields:

    param_i.fold1train  score
    param_i.fold1test   score
    param_i.fold2train  score
    param_i.fold2test   score
    ...
    param_j.fold1train  score
    param_j.fold1test   score
    param_j.fold2train  score
    param_j.fold2test   score

    do k-fold cross validation and report mean test score.
    """
    scores = defaultdict(dict)
    with open(tsv_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            name = fields[name_field]
            if name == '':
                continue
            try:
                score = float(fields[score_field])
            except ValueError:
                #if verbose:
                #    print('skip this line:', line)
                continue
            if postfix is None:
                m = re.match(r'(.*)fold([0-9]+)(train|test)$', name)
            else:
                m = re.match(
                    r'(.*)fold([0-9]+)(train|test)' + postfix + '$', name)
            if m is None:
                continue
            params, k, kind = m.group(1), m.group(2), m.group(3)
            if kind == 'test': # test score
                scores[k]['__test__' + params] = score
            elif kind == 'train': # validation score
                scores[k][params] = score
            else:
                assert 0, f"unexpected: {kind}"
    assert len(scores) != 0

    test_scores = []
    best_params_set = defaultdict(int)
    for k, score_dict in scores.items():
        tune_scores = list(filter(
            lambda x: not x[0].startswith('__test__'),
            score_dict.items() # [(params, score), ...]
        ))
        best_params_idx = max(range(len(tune_scores)), key=lambda i: tune_scores[i][1])
        best_params = tune_scores[best_params_idx][0]
        test_score = score_dict['__test__' + best_params]
        #if verbose:
        #    print(f'fold#{k}: test_score={test_score}')
        test_scores.append(test_score)
        best_params_set[best_params] += 1
    mean_test_score = np.array(test_scores).mean()
    mean_test_score = round(mean_test_score, 4)
    if verbose:
        print('best params and frqs:', best_params_set.items())
        print('mean test score:', mean_test_score)
    else:
        print(mean_test_score)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'split_run_files': split_run_files,
        'cross_validate_tsv': cross_validate_tsv,
    })
