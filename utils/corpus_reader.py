import os


def file_iterator(corpus, endat, ext):
    cnt = 0
    for dirname, dirs, files in os.walk(corpus):
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


def corpus_length__ntcir12_txt(latex_list_file, max_length):
    with open(latex_list_file) as f:
        n_lines = sum(1 for _ in f)
    return n_lines if max_length == 0 else min(n_lines, max_length)


def corpus_reader__ntcir12_txt(latex_list_file):
    with open(latex_list_file, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            docid_and_pos = fields[0]
            latex = ' '.join(fields[1:])
            latex = latex.replace('% ', '')
            latex = f'[imath]{latex}[/imath]'
            yield docid_and_pos, latex # docid, contents


def corpus_length__arqmath3_rawxml(xml_file, max_length):
    from xmlr import xmliter
    cnt = 0
    for attrs in xmliter(xml_file, 'row'):
        if cnt + 1 > max_length:
            return max_length
        cnt += 1
    return cnt


def corpus_reader__arqmath3_rawxml(xml_file):
    from xmlr import xmliter
    from bs4 import BeautifulSoup
    from replace_post_tex import replace_dollar_tex
    def html2text(html):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            elem.replace_with('[imath]' + elem.text + '[/imath]')
        return soup.text
    def comment2text(html):
        soup = BeautifulSoup(html, "html.parser")
        return replace_dollar_tex(soup.text)

    if 'Posts' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Body' not in attrs:
                body = None
            else:
                body = html2text(attrs['@Body'])
            ID = attrs['@Id']
            vote = attrs['@Score']
            postType = attrs['@PostTypeId']
            if postType == "1": # Question
                title = html2text(attrs['@Title'])
                tags = attrs['@Tags']
                tags = tags.replace('-', '_')
                if '@AcceptedAnswerId' in attrs:
                    accept = attrs['@AcceptedAnswerId']
                else:
                    accept = None
                yield 'Q', ID, title, body, vote, tags, accept
            else:
                assert postType == "2" # Answer
                parentID = attrs['@ParentId']
                yield 'A', ID, parentID, body, vote

    elif 'Comments' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Text' not in attrs:
                comment = None
            else:
                comment = comment2text(attrs['@Text'])
            ID = attrs['@Id']
            answerID = attrs['@PostId']
            yield 'C', ID, answerID, comment
    else:
        raise NotImplemented


def corpus_length__arqmath_answer(corpus_dir, max_length):
    print('counting answer files:', corpus_dir)
    return sum(1 for _ in
        file_iterator(corpus_dir, max_length, 'answer')
    )


def corpus_reader__arqmath_answer(corpus_dir):
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'answer'):
        path = dirname + '/' + fname
        content = file_read(path)
        fields = os.path.basename(path).split('.')
        A_id, Q_id = int(fields[0]), int(fields[1])
        yield A_id, content # docid, contents


def corpus_length__arqmath_task2_tsv(corpus_dir, max_length):
    print('counting tsv file lengths:', corpus_dir)
    cnt = 0
    for _, dirname, fname in file_iterator(corpus_dir, max_length, 'tsv'):
        path = dirname + '/' + fname
        with open(path, 'r') as fh:
            n_lines = sum(1 for _ in fh)
        cnt += n_lines
        print(fname, n_lines)
    return cnt


def corpus_reader__arqmath_task2_tsv(corpus_dir):
    import csv
    import html
    from collections import defaultdict

    visual_id_cnt = defaultdict(lambda: 0)
    for cnt, dirname, fname in file_iterator(corpus_dir, -1, 'tsv'):
        path = dirname + '/' + fname
        with open(path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if i == 0:
                    yield None
                    continue
                formulaID = line[0]
                post_id = line[1]
                thread_id = line[2]
                type_ = line[3] # 'question,' 'comment,' 'answer,' or 'title.'
                if type_ == 'comment':
                    yield None
                    continue
                visual_id = line[4]
                latex = html.unescape(line[5])
                if visual_id_cnt[visual_id] >= 5:
                    yield None
                    continue
                else:
                    visual_id_cnt[visual_id] += 1
                latex = f'[imath]{latex}[/imath]'
                yield (formulaID, post_id), latex # docid, contents

