import os
import fire
import pickle
import random
from tqdm import tqdm

import _pya0
from preprocess import preprocess_for_transformer


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


def generate_contrastive_pairs(
    q_dict_file='arqmath-question-dict.pkl',
    a_dict_file='arqmath-answer-dict.pkl',
    tag_bank_file='arqmath-tag-bank.pkl',
    answer_bank_file='arqmath-answer-bank.pkl',
    n_splits=4, limit=-1, debug=False,
    random_seed=123, allow_vote_postive=False):

    random.seed(random_seed)

    q_dict = load_pickle_file(q_dict_file)
    a_dict = load_pickle_file(a_dict_file)
    tag_bank = load_pickle_file(tag_bank_file)
    if allow_vote_postive:
        answer_bank = load_pickle_file(answer_bank_file)

    all_questions = list(q_dict.items())
    if limit >= 0: all_questions = all_questions[:limit]

    aggregate = []
    aggregate_cnt = 0
    n_per_split = len(all_questions) // n_splits
    with tqdm(all_questions) as progress:
        for qid, (ac, tags, Q) in progress:
            reminder = aggregate_cnt % n_per_split
            progress.set_description(f'{reminder} % {n_per_split}')
            # retrieve postive
            if allow_vote_postive:
                positives = filter(lambda x: x[1] > 0, answer_bank[qid])
                positives = [a_dict[p[0]] for p in positives]
            else:
                if ac not in a_dict:
                    continue
                positives = [a_dict[ac]]
            Q = preprocess_for_transformer(Q)

            for positive_A in positives:
                # sample negative
                tag = random.choice(tags)
                negative_id = random.choice(tag_bank[tag])
                if negative_id not in a_dict:
                    print(f'Warning: neg answer #{negative_id} not found!')
                    continue
                negative_A = a_dict[negative_id]

                if debug:
                    print('----\n')
                    print(Q, end='\n\n')
                    print(positive_A, end='\n\n')
                    print(f'[{tag}]')
                    print(negative_A, end='\n\n')
                    quit(0)

                positive_A = preprocess_for_transformer(positive_A)
                negative_A = preprocess_for_transformer(negative_A)

                aggregate.append((Q, tag, positive_A, negative_A))
                aggregate_cnt += 1
                if aggregate_cnt % n_per_split == 0:
                    aggregate = dump_split(aggregate, aggregate_cnt)

        dump_split(aggregate, aggregate_cnt)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(generate_contrastive_pairs)
