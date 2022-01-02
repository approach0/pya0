import os
import fire
import pickle
import random
from tqdm import tqdm
from collections import defaultdict

import _pya0
from preprocess import preprocess_for_transformer

from xmlr import xmliter


def load_pickle_file(file):
    with open(file, 'rb') as fh:
        print(f'Loading {file} ...')
        return pickle.load(fh)


def dump_split(aggregate, aggregate_cnt, output_dir='./data'):
    output_file = f'{output_dir}/qa.pairs.{aggregate_cnt}'
    with open(output_file, 'wb') as fh:
        L = len(aggregate)
        print(f'writing split {output_file} of length {L} ...')
        pickle.dump(aggregate, fh)
    return []


def read_linked_posts(postlink_file):
    dups_dict = defaultdict(list)
    for attrs in xmliter(postlink_file, 'row'):
        if attrs['@LinkTypeId'] != '3':
            continue # skip weakly relevant ones (linked posts)
        a = int(attrs['@PostId'])
        b = int(attrs['@RelatedPostId'])
        dups_dict[a].append(b)
        dups_dict[b].append(a)
    return dups_dict


def sample_hard_negative(tags, tag_bank, a_dict):
    if len(tags) == 0:
        tag = random.choice(tag_bank.items())
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
    postlink_file='PostLinks.V1.2.xml',
    n_splits=10, limit=-1, debug=False, min_votes=7,
    random_seed=123, allow_vote_postive=True):

    print(f'Reading {postlink_file} ...')
    dups_dict = read_linked_posts(postlink_file)

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
            Q = preprocess_for_transformer(Q)

            # retrieve thread AC as postive
            if ac not in a_dict:
                continue
            positives = [a_dict[ac]]
            # add highly upvoted answers as additional postives
            if allow_vote_postive:
                answers = filter(lambda x: x[1] >= min_votes, answer_bank[qid])
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

            for positive_A in positives:
                all_tags = tags + dup_tags # all relevant tags
                negative_A = sample_hard_negative(all_tags, tag_bank, a_dict)
                if negative_A is None:
                    print(f'Warning: neg answer #{negative_id} not found!')
                    continue

                if debug:
                    print('----\n')
                    print(Q, end='\n\n')
                    print(positive_A, end='\n\n')
                    print(negative_A, end='\n\n')
                    quit(0)

                positive_A = preprocess_for_transformer(positive_A)
                negative_A = preprocess_for_transformer(negative_A)

                aggregate.append((Q, all_tags, positive_A, negative_A))
                aggregate_cnt += 1
                if aggregate_cnt % n_per_split == 0:
                    aggregate = dump_split(aggregate, aggregate_cnt)

        dump_split(aggregate, aggregate_cnt)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_contrastive_pairs)
