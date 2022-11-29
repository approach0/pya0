import pickle
from tqdm import tqdm
from corpus_reader import *
from collections import defaultdict


def generate_qdict(xml_file, maxitems=0):
    total_lines = corpus_length__arqmath3_rawxml(xml_file, maxitems)
    print('total lines:', total_lines)
    reader = corpus_reader__arqmath3_rawxml(xml_file)

    Q_dict = dict()
    for cnt, (meta_data, data) in tqdm(enumerate(reader), total=total_lines):
        if cnt >= total_lines: break
        type_ = meta_data[1]
        if type_ == 'Q':
            ID, _, title, body, vote, tags, accept = meta_data
            assert isinstance(ID, str)
            assert isinstance(vote, str)
            assert isinstance(accept, str) or accept is None
            tags = tags.replace('<', ' ').replace('>', '').split()
            Q_content = title + '\n\n' + body
            Q_dict[ID] = (accept, tags, Q_content)
        else:
            continue

    with open('arqmath-question-dict.pkl', 'wb') as fh:
        pickle.dump(Q_dict, fh)


def generate_answer_banks(xml_file, maxitems=0):
    with open('arqmath-question-dict.pkl', 'rb') as fh:
        Q_dict = pickle.load(fh)

    total_lines = corpus_length__arqmath3_rawxml(xml_file, maxitems)
    print('total lines:', total_lines)
    reader = corpus_reader__arqmath3_rawxml(xml_file)

    A_dict = dict()
    tag_dict = defaultdict(list)
    answer_bank = defaultdict(list)
    for cnt, (meta_data, data) in tqdm(enumerate(reader), total=total_lines):
        if cnt >= total_lines: break
        type_ = meta_data[1]
        if type_ == 'A':
            ID, _, parentID, vote = meta_data
            assert isinstance(ID, str)
            assert isinstance(parentID, str)
            assert isinstance(vote, str)
            if parentID not in Q_dict:
                print(f'{parentID} not in Q_dict.')
                continue
            A_dict[ID] = data
            _, tags, _ = Q_dict[parentID]
            for tag in tags:
                tag_dict[tag].append(ID)
            answer_bank[parentID].append((ID, vote))
        else:
            continue

    # A_id -> A
    with open('arqmath-answer-dict.pkl', 'wb') as fh:
        pickle.dump(A_dict, fh)
    # tag -> [A_id, ...]
    with open('arqmath-tag-bank.pkl', 'wb') as fh:
        pickle.dump(tag_dict, fh)
    # Q_id -> [(A_id, upvotes), ...]
    with open('arqmath-answer-bank.pkl', 'wb') as fh:
        pickle.dump(answer_bank, fh)


if __name__ == '__main__':
    import os
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'gen_question_dict': generate_qdict,
        'gen_answer_banks': generate_answer_banks
    })
