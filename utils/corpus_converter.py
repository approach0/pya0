import os
import re
import fire
import json
from tqdm import tqdm
from corpus_reader import *
from collections import defaultdict


def convert_ntcir12_wfb_to_jsonl(corpus_file,
    output_file='ntcir12_wfb.jsonl',
    max_items=float('inf')):
    n = corpus_length__ntcir12_txt(corpus_file, max_items)
    print(f'{n} formulas in total.')
    reader = corpus_reader__ntcir12_txt(corpus_file)
    progress = tqdm(reader, total=n)
    with open(output_file, 'w') as fh:
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            (formulaID, ), latex = row
            fh.write(json.dumps({
                'formulaID': formulaID,
                'latex': latex
            }, sort_keys=True))
            fh.write('\n')


def add_arqmath_task1_row(questions, answers, row):
    if row[0][1] == 'Q':
        (thread_id, _, title, question, vote, tags, accept), __ = row
        if question is None:
            return
        questions[thread_id]['title'] = title
        questions[thread_id]['question'] = question
        questions[thread_id]['question_vote'] = vote
        questions[thread_id]['tags'] = tags
        questions[thread_id]['accept'] = accept
    elif row[0][1] == 'A':
        (answer_id, _, thread_id, vote), answer = row
        if answer is None:
            return
        answers[answer_id]['thread_id'] = thread_id
        answers[answer_id]['answer'] = answer
        answers[answer_id]['answer_vote'] = vote
    elif row[0][1] == 'C':
        (answer_id, _, comment_id, comment), __ = row
        if comment is None:
            return
        answers[answer_id]['comment__' + comment_id] = comment
    else:
        raise NotImplemented


def convert_arqmath_task1_to_jsonl(corpus_dir,
    post_file='Posts.V1.3.xml',
    comment_file='Comments.V1.0.xml',
    output_file='arqmath_task1.jsonl',
    max_items=float('inf')):
    post_file = os.path.join(corpus_dir, post_file)
    comment_file = os.path.join(corpus_dir, comment_file)

    Q = defaultdict(dict)
    A = defaultdict(dict)
    with open(output_file, 'w') as fh:
        print('Reading', post_file)
        n = corpus_length__arqmath3_rawxml(post_file, max_items)
        reader = corpus_reader__arqmath3_rawxml(post_file)
        progress = tqdm(reader, total=n)
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            add_arqmath_task1_row(Q, A, row)

        print('Reading', comment_file)
        n = corpus_length__arqmath3_rawxml(comment_file, max_items)
        reader = corpus_reader__arqmath3_rawxml(comment_file)
        progress = tqdm(reader, total=n)
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            add_arqmath_task1_row(Q, A, row)

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


def convert_arqmath_task2_to_jsonl(corpus_dir,
    output_file='arqmath_task2.jsonl',
    max_items=float('inf')):
    n = corpus_length__arqmath_task2_tsv(corpus_dir, max_items)
    print(f'{n} formulas in total.')
    reader = corpus_reader__arqmath_task2_tsv(corpus_dir)
    progress = tqdm(reader, total=n)
    with open(output_file, 'w') as fh:
        for idx, row in enumerate(progress):
            if idx >= n:
                break
            elif row[1] is None:
                continue
            (formulaID, *doc_props), latex = row
            fh.write(json.dumps({
                'formulaID': formulaID,
                'doc_props': doc_props,
                'latex': latex
            }, sort_keys=True))
            fh.write('\n')


def convert_arqmath_contextual_task2_to_jsonl(corpus_dir,
    post_file='Posts.V1.3.xml',
    task2_jsonl='arqmath3_task2.jsonl', # to ensure cnt(visual_id) <= 5
    output_file='arqmath_contextual_task2.jsonl',
    max_items=float('inf')):
    post_file = os.path.join(corpus_dir, post_file)
    task2_jsonl = os.path.join(corpus_dir, task2_jsonl)

    total_n = corpus_length__jsonl(task2_jsonl, None, 0)
    reader = corpus_reader__jsonl(task2_jsonl, "['formulaID']")
    progress = tqdm(reader, total=total_n)
    print('Collecting visually distinct formula IDs...')
    allow_formulaIDs = set()
    for _, formulaID in progress:
        allow_formulaIDs.add(formulaID)

    with open(output_file, 'w') as fh:
        print('Reading', post_file)
        total_n = corpus_length__arqmath3_rawxml(post_file, max_items)
        reader = corpus_reader__arqmath3_rawxml(post_file,
            preserve_formula_ids=True)
        progress = tqdm(reader, total=total_n)
        for idx, row in enumerate(progress):
            (postID, _type, *_), body = row # task2 allow searching for both Q or A.
            if idx >= total_n:
                break
            elif _type == 'A':
                pass
            elif _type == 'Q':
                title = row[0][2]
                qbody = row[0][3]
                body = title + '\n' + qbody
                pass
            else:
                continue
            if '[/imath]' in body:
                matches = re.finditer(
                    r'\[imath id="(\d+)"\](.*?)\[/imath\]', body
                )
                matches = list(reversed([m for m in matches]))
                for i, n in enumerate(matches):
                    formulaID, tex = n.group(1), n.group(2)
                    formula_len = len(tex.strip())
                    if formulaID not in allow_formulaIDs or formula_len <= 2:
                        continue
                    ctx = body[:]
                    for j, m in enumerate(matches):
                        span, _, tex = m.span(), m.group(1), m.group(2)
                        if i == j:
                            wrap = f'[imath]{tex}[/imath]'
                            ctx = ctx[:span[0]] + wrap + ctx[span[1]:]
                        else:
                            ctx = ctx[:span[0]] + '[MASK]' + ctx[span[1]:]
                    fh.write(json.dumps({
                        'formulaID': formulaID,
                        'doc_props': (postID, _type, str(i)),
                        'latex': ctx
                    }, sort_keys=True))
                    fh.write('\n')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'ntcir12_wfb': convert_ntcir12_wfb_to_jsonl,
        'arqmath_task1': convert_arqmath_task1_to_jsonl,
        'arqmath_task2': convert_arqmath_task2_to_jsonl,
        'arqmath_contextual_task2': convert_arqmath_contextual_task2_to_jsonl
    })
