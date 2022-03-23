import os
import fire
import json
from tqdm import tqdm
from corpus_reader import *
from collections import defaultdict


def add_arqmath_row(questions, answers, row):
    if row[0] == 'Q':
        _, thread_id, title, question, vote, tags, accept = row
        if question is None:
            return
        questions[thread_id]['title'] = title
        questions[thread_id]['question'] = question
        questions[thread_id]['question_vote'] = vote
        questions[thread_id]['tags'] = tags
        questions[thread_id]['accept'] = accept
    elif row[0] == 'A':
        _, answer_id, thread_id, answer, vote = row
        if answer is None:
            return
        answers[answer_id]['thread_id'] = thread_id
        answers[answer_id]['answer'] = answer
        answers[answer_id]['answer_vote'] = vote
    elif row[0] == 'C':
        _, comment_id, answer_id, comment = row
        if comment is None:
            return
        answers[answer_id]['comment__' + comment_id] = comment
    else:
        raise NotImplemented


def convert_arqmath_task1_to_jsonl(corpus_dir,
    post_file='Posts.V1.3.xml',
    comment_file='Comments.V1.0.xml',
    output_file='arqmath.jsonl',
    max_length=float('inf')):
    post_file = os.path.join(corpus_dir, post_file)
    comment_file = os.path.join(corpus_dir, comment_file)

    Q = defaultdict(dict)
    A = defaultdict(dict)
    with open(output_file, 'w') as fh:
        print('Reading', post_file)
        n = corpus_length__arqmath3_rawxml(post_file, max_length)
        reader = corpus_reader__arqmath3_rawxml(post_file)
        progress = tqdm(reader, total=n)
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            add_arqmath_row(Q, A, row)

        print('Reading', comment_file)
        n = corpus_length__arqmath3_rawxml(comment_file, max_length)
        reader = corpus_reader__arqmath3_rawxml(comment_file)
        progress = tqdm(reader, total=n)
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            add_arqmath_row(Q, A, row)

        print('Generating', output_file)
        for answer_id, answer_dict in A.items():
            if 'thread_id' not in answer_dict:
                continue
            thread_id = answer_dict['thread_id']
            if thread_id not in Q:
                print(f'Warning: thread#{thread_id} not found.')
                continue
            question_dict = Q[thread_id]
            merged_dict = {**question_dict, **answer_dict}
            comment_keys = list(filter(lambda x: x.startswith('comment__'),
                merged_dict.keys()))
            if len(comment_keys) > 0:
                comments = list(map(lambda x: merged_dict[x], comment_keys))
                comments = '\n\n'.join(comments)
            else:
                comments = None
            for key in merged_dict.copy():
                if key.startswith('comment__'):
                    del merged_dict[key]
            merged_dict['comments'] = comments
            merged_dict['thread_id'] = thread_id
            merged_dict['url'] = \
                f'https://math.stackexchange.com/questions/{thread_id}/'
            merged_dict['answer_id'] = answer_id
            fh.write(json.dumps(merged_dict, sort_keys=True))
            fh.write('\n')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'arqmath_task1': convert_arqmath_task1_to_jsonl
    })
