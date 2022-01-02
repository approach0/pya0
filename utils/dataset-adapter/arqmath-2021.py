import os
import fire
import json
import pickle
from collections import defaultdict


def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        print(dirname)
        for f in files:
            if cnt >= endat and endat > 0:
                return
            elif f.split('.')[-1] == ext:
                cnt += 1
                yield (cnt, dirname, f)


def file_read(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as fh:
        return fh.read()


def generate_qdict(corpus, endat=-1):
    Q_dict = dict()
    for cnt, dirname, fname in file_iterator(corpus, endat, 'question'):
        path = dirname + '/' + fname
        Q_id = os.path.basename(path).split('.')[0]
        Q = file_read(path)
        ac = file_read(path + '_ac')
        tags = file_read(path + '_tags')
        tags = tags.replace('<', ' ').replace('>', '').split()
        if ac is None:
            continue
        Q_dict[int(Q_id)] = (int(ac), tags, Q)

    # Q_id -> (ac, tags, Q)
    with open('arqmath-question-dict.pkl', 'wb') as fh:
        pickle.dump(Q_dict, fh)


def generate_tag_bank(corpus, endat=-1):
    with open('arqmath-question-dict.pkl', 'rb') as fh:
        Q_dict = pickle.load(fh)

    A_dict = dict()
    tag_dict = defaultdict(list)
    for cnt, dirname, fname in file_iterator(corpus, endat, 'answer'):
        path = dirname + '/' + fname
        A = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id = int(fields[0])
        Q_id = int(fields[1])

        if Q_id not in Q_dict:
            continue

        A_dict[A_id] = A
        _, tags, _ = Q_dict[Q_id]
        for tag in tags:
            tag_dict[tag].append(A_id)

    # A_id -> A
    with open('arqmath-answer-dict.pkl', 'wb') as fh:
        pickle.dump(A_dict, fh)
    # tag -> [A_id, ...]
    with open('arqmath-tag-bank.pkl', 'wb') as fh:
        pickle.dump(tag_dict, fh)


def generate_answer_bank(corpus, endat=-1):
    with open('arqmath-question-dict.pkl', 'rb') as fh:
        Q_dict = pickle.load(fh)

    answer_bank = defaultdict(list)
    for cnt, dirname, fname in file_iterator(corpus, endat, 'answer'):
        path = dirname + '/' + fname
        A = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id = int(fields[0])
        Q_id = int(fields[1])

        if Q_id not in Q_dict:
            continue

        vote_file = f'{dirname}/{A_id}.answer_vote'
        upvotes = int(file_read(vote_file))
        answer_bank[Q_id].append((A_id, upvotes))

    # Q_id -> [(A_id, upvotes), ...]
    with open('arqmath-answer-bank.pkl', 'wb') as fh:
        pickle.dump(answer_bank, fh)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'question_dict': generate_qdict,
        'tag_bank': generate_tag_bank,
        'answer_bank': generate_answer_bank
    })
