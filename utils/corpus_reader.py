import os


def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
        for f in sorted(files):
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


def corpus_length__ntcir12_txt(latex_list_file, max_items):
    with open(latex_list_file) as f:
        n_lines = sum(1 for _ in f)
    return n_lines if max_items == 0 else min(n_lines, max_items)


def corpus_reader__ntcir12_txt(latex_list_file):
    with open(latex_list_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'
            # YIELD (docid, doc_props), contents
            yield (docid_and_pos,), latex


def corpus_length__arqmath3_rawxml(xml_file, max_items):
    from xmlr import xmliter
    cnt = 0
    for attrs in xmliter(xml_file, 'row'):
        if cnt + 1 > max_items and max_items > 0:
            return max_items
        cnt += 1
    return cnt


def corpus_reader__arqmath3_rawxml(xml_file, preserve_formula_ids=False):
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    from replace_post_tex import replace_dollar_tex
    def html2text(html, preserve):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            if not preserve:
                elem.replace_with('[imath]' + elem.text + '[/imath]')
            else:
                formula_id = elem.get('id')
                if formula_id is None:
                    elem.replace_with(' ')
                else:
                    elem.replace_with(
                        f'[imath id="{formula_id}"]' + elem.text + '[/imath]'
                    )
        return soup.text
    def comment2text(html):
        soup = BeautifulSoup(html, "html.parser")
        return replace_dollar_tex(soup.text)

    if 'Posts' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Body' not in attrs:
                body = None
            else:
                body = html2text(attrs['@Body'], preserve_formula_ids)
            ID = attrs['@Id']
            vote = attrs['@Score']
            postType = attrs['@PostTypeId']
            if postType == "1": # Question
                title = html2text(attrs['@Title'], preserve_formula_ids)
                tags = attrs['@Tags']
                tags = tags.replace('-', '_')
                if '@AcceptedAnswerId' in attrs:
                    accept = attrs['@AcceptedAnswerId']
                else:
                    accept = None
                # YIELD (docid, doc_props), contents
                yield (ID, 'Q', title, body, vote, tags, accept), None
            else:
                assert postType == "2" # Answer
                parentID = attrs['@ParentId']
                # YIELD (docid, doc_props), contents
                yield (ID, 'A', parentID, vote), body

    elif 'Comments' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Text' not in attrs:
                comment = None
            else:
                comment = comment2text(attrs['@Text'])
            ID = attrs['@Id']
            answerID = attrs['@PostId']
            # YIELD (docid, doc_props), contents
            yield (answerID, 'C', ID, comment), None
    else:
        raise NotImplemented


# this function only **estimate** number of items by line numbers
def corpus_length__arqmath_task2_tsv(corpus_dir, max_items):
    print('counting tsv file lengths:', corpus_dir)
    cnt = 0
    for _, dirname, fname in file_iterator(corpus_dir, 0, 'tsv'):
        path = dirname + '/' + fname
        with open(path, 'r') as fh:
            n_lines = sum(1 for _ in fh)
        cnt += n_lines
        if cnt >= max_items and max_items > 0:
            cnt = max_items
            break
        print(fname, n_lines)
    return cnt


def corpus_reader__arqmath_task2_tsv(corpus_dir):
    import csv
    import html
    from collections import defaultdict

    visual_id_cnt = defaultdict(lambda: 0)
    guess_version = None
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'tsv'):
        path = dirname + '/' + fname
        with open(path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, fields in enumerate(tsvreader):
                if i == 0:
                    header = ' '.join(fields)
                    if 'old_visual_id' in header:
                        guess_version = 'v3'
                    else:
                        guess_version = 'v2'
                    # YIELD (docid, doc_props), contents
                    yield None, None
                    continue
                # determine type in {question, comment, answer, title}
                type_ = fields[3]
                if type_ == 'comment':
                    # YIELD (docid, doc_props), contents
                    yield None, None
                    continue
                if guess_version == 'v2':
                    #id, post_id, thread_id, type, visual_id, formula.
                    visual_id = fields[4]
                elif guess_version == 'v3':
                    # id, post_id, thread_id, type, comment_id,
                    # old_visual_id, visual_id, issue, formula.
                    visual_id = fields[6]
                else:
                    assert 0, 'guess_version not handled.'
                if visual_id_cnt[visual_id] >= 5:
                    # YIELD (docid, doc_props), contents
                    yield None, None
                    continue
                else:
                    visual_id_cnt[visual_id] += 1
                formulaID = fields[0]
                doc_props = fields[1:-1]
                latex = html.unescape(fields[-1])
                latex = f'[imath]{latex}[/imath]'
                # YIELD (docid, doc_props), contents
                yield (formulaID, *doc_props), latex


def corpus_length__jsonl(jsonl_path, fields, max_items):
    return corpus_length__ntcir12_txt(jsonl_path, max_items)


def corpus_reader__jsonl(jsonl_path, fields):
    import json
    import pdb
    fields = eval(fields)
    with open(jsonl_path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            j = json.loads(line)
            values = [j[f] for f in fields]
            # YIELD (docid, doc_props), contents
            yield tuple(values[:-1]), values[-1]
