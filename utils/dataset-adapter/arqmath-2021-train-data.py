import os
import fire
import json
import pickle
import random
from tqdm import tqdm
from collections import defaultdict

import _pya0
import preprocess

from xmlr import xmliter


def load_pickle_file(file):
    with open(file, 'rb') as fh:
        print(f'Loading {file} ...')
        return pickle.load(fh)


def dump_split(out_dir, aggregate, aggregate_cnt, ver):
    output_file = os.path.join(out_dir, f'qa-v{ver}.pairs.{aggregate_cnt}')
    with open(output_file, 'wb') as fh:
        L = len(aggregate)
        print(f'writing split {output_file} of length {L} ...')
        pickle.dump(aggregate, fh)
    return []


def read_dup_posts(postlink_file):
    dups_dict = defaultdict(list)
    for attrs in xmliter(postlink_file, 'row'):
        if attrs['@LinkTypeId'] != '3':
            continue # skip weakly relevant ones (linked posts)
        a = str(attrs['@PostId'])
        b = str(attrs['@RelatedPostId'])
        dups_dict[a].append(b)
        dups_dict[b].append(a)
    return dups_dict


def sample_hard_negative(tags, tag_bank, a_dict):
    if len(tags) == 0:
        tag = random.choice(list(tag_bank.keys()))
    else:
        tag = random.choice(tags)
    negative_id = random.choice(tag_bank[tag])
    if negative_id not in a_dict:
        return None
    return a_dict[negative_id]


def generate_contrastive_pairs(
    q_dict_file='arqmath-question-dict.pkl',
    a_dict_file='arqmath-answer-dict.pkl',
    tag_bank_file='arqmath-tag-bank.pkl',
    answer_bank_file='arqmath-answer-bank.pkl',
    postlink_file='PostLinks.V1.2.xml', dest_token='pya0',
    replace_isolated_groups=True, out_dir='.', print_frq=10_000,
    num_tokenizer_ver=3, n_splits=10, limit=-1, min_votes=7,
    random_seed=123, allow_vote_postive=True):

    print(f'Reading {postlink_file} ...')
    dups_dict = read_dup_posts(postlink_file)

    print('Reading pickle files ...')
    random.seed(random_seed)
    q_dict = load_pickle_file(q_dict_file)
    a_dict = load_pickle_file(a_dict_file)
    tag_bank = load_pickle_file(tag_bank_file)
    if allow_vote_postive:
        answer_bank = load_pickle_file(answer_bank_file)

    print('Generating Q&A pairs')
    all_questions = list(q_dict.items())
    if limit >= 0: all_questions = all_questions[:limit]

    aggregate = []
    aggregate_cnt = 0
    n_per_split = len(all_questions) // n_splits
    with tqdm(all_questions) as progress:
        for qid, (ac, tags, Q) in progress:
            reminder = aggregate_cnt % n_per_split
            progress.set_description(f'{reminder} % {n_per_split}')

            Q = preprocess.preprocess_for_transformer(Q,
                num_tokenizer_ver=num_tokenizer_ver,
                replace_isolated_groups=replace_isolated_groups,
                dest_token=dest_token
            )

            positives = []
            # retrieve thread AC as postive
            if ac in a_dict:
                positives.append(a_dict[ac])
            # add highly upvoted answers as additional postives
            if allow_vote_postive:
                answers = filter(
                    lambda x: int(x[1]) >= min_votes, answer_bank[qid]
                )
                positives += [a_dict[a[0]] for a in answers if a[0] != ac]

            # any dup-thread Q&A ?
            dup_tags = []
            if len(dups_dict[qid]) > 0:
                for dup_qid in dups_dict[qid]:
                    if dup_qid not in q_dict:
                        continue
                    dup_ac, dup_tags, dup_Q = q_dict[dup_qid]
                    #print('sample duplicate question as negative')
                    positives.append(dup_Q)
                    if dup_ac not in a_dict:
                        continue
                    #print('sample duplicate question AC as negative')
                    positives.append(a_dict[dup_ac])

            for i, positive_A in enumerate(positives):
                all_tags = tags + dup_tags # all relevant tags
                negative_A = sample_hard_negative(all_tags, tag_bank, a_dict)
                if negative_A is None:
                    print(f'Warning: neg answer #{negative_id} not found!')
                    continue

                positive_A = preprocess.preprocess_for_transformer(positive_A,
                    num_tokenizer_ver=num_tokenizer_ver,
                    replace_isolated_groups=replace_isolated_groups,
                    dest_token=dest_token
                )
                negative_A = preprocess.preprocess_for_transformer(negative_A,
                    num_tokenizer_ver=num_tokenizer_ver,
                    replace_isolated_groups=replace_isolated_groups,
                    dest_token=dest_token
                )

                if i == 0 and aggregate_cnt % print_frq == 0:
                    print('=' * 50)
                    print(Q)
                    print('-' * 50)
                    print(positive_A)
                    print('~' * 50)
                    print(negative_A, end='\n\n')

                aggregate.append((Q, all_tags, positive_A, negative_A))
                aggregate_cnt += 1
                if aggregate_cnt % n_per_split == 0:
                    aggregate = dump_split(out_dir,
                        aggregate, aggregate_cnt, num_tokenizer_ver
                    )

        dump_split(out_dir,
            aggregate, aggregate_cnt, num_tokenizer_ver
        )


def generate_duplicate_posts(
    q_dict_file='arqmath-question-dict.pkl',
    a_dict_file='arqmath-answer-dict.pkl',
    answer_bank_file='arqmath-answer-bank.pkl',
    postlink_file='PostLinks.V1.3.xml'):

    q_dict = load_pickle_file(q_dict_file)
    a_dict = load_pickle_file(a_dict_file)
    dup_dicts = read_dup_posts(postlink_file)
    coverage = set()
    fh = open('arqmath-question-dups.jsonl', 'w')

    for qid, dups in dup_dicts.items():
        if len(dups) > 0:
            for dup_qid in dups:
                if qid not in q_dict or dup_qid not in q_dict:
                    continue
                _, _, Q = q_dict[qid]
                _, _, Q_dup = q_dict[dup_qid]
                if '[imath]' not in Q: continue
                if '[imath]' not in Q_dup: continue
                if 'Possible Duplicate' in Q: continue
                if 'Possible Duplicate' in Q_dup: continue
                if qid in coverage: continue
                if dup_qid in coverage: continue
                fh.write(json.dumps({
                    'qid': qid,
                    'Q': Q,
                    'dup_qid': dup_qid,
                    'Q_dup': Q_dup
                }) + '\n')
                coverage.add(qid)
                coverage.add(dup_qid)
    fh.close()


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'gen_pairs': generate_contrastive_pairs,
        'gen_dups': generate_duplicate_posts
    })
