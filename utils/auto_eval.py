import subprocess
import csv
import os
import numpy as np
import itertools
import math


def remake(root_dir):
    res = subprocess.run(['make'], cwd=f'{root_dir}')
    if 0 != res.returncode:
        print('Error: make non-zero return value!')
        quit(1)
    res = subprocess.run(['make'], cwd=f'{root_dir}/pya0')
    if 0 != res.returncode:
        print('Error: make non-zero return value!')
        quit(1)


def _read_templates(directory):
    templates = []
    for _, _, file_names in os.walk(directory):
        for file_name in file_names:
            if file_name.startswith('__link__'):
                continue
            template = {}
            with open(f'{directory}/{file_name}') as fh:
                template['txt'] = fh.read()
            with open(f'{directory}/__link__{file_name}') as fh:
                template['output'] = fh.read().strip('\n')
            templates.append(template)
    return templates


def replace_source_code(templates_dir, replaces):
    templates = _read_templates(templates_dir)
    for k, v in replaces.items():
        for t in templates:
            t['txt'] = t['txt'].replace('{{' + k + '}}', v)
            print(f'{t["output"]}: replace "{k}" --> "{v}"')
    for t in templates:
        with open(t['output'], 'w') as f:
            f.write(t['txt'])


def tsv_product(input_tsv):
    output = '#'
    with open(input_tsv) as fd:
        tot_rows = sum(1 for _ in open(input_tsv))
        rd = csv.reader(fd, delimiter="\t")
        entries = []
        matrix = []
        for idx, row in enumerate(rd):
            entry = row[0].strip() if len(row) > 0 else ''
            if entry[0] == '#':
                continue # skip comment
            values = list(filter(lambda x: len(x) > 0, row[1:]))
            assert len(values) > 0
            if values[0] ==  '<range>':
                start, end, step = map(lambda x: float(x), values[1:])
                values = np.arange(start, end, step).tolist()
                values = map(lambda x: str(x), values)
            entries.append(entry)
            matrix.append(values)
        prods = list(itertools.product(*matrix))
        output += '\t'.join(entries) + '\n'
        for p in prods:
            output += '\t'.join(p) + '\n'
    return output.strip('\n')


def tsv_eval_read(input_tsv):
    with open(input_tsv) as fd:
        tot_rows = sum(1 for _ in open(input_tsv))
        rd = csv.reader(fd, delimiter="\t")
        header = []
        rows = []
        for idx, row in enumerate(rd):
            entry = row[0].strip() if len(row) > 0 else ''
            if entry[0] == '#':
                row[0] = entry.strip('#')
                header = row
                continue # skip header
            rows.append(row)
    return header, rows


def tsv_eval_do(header, rows, each_do, prefix=''):
    tot_rows = len(rows)
    for idx, row in enumerate(rows):
        run_name = prefix + ','.join([f'{k}={v}' for k,v in zip(header, row)])
        print('[row %u / %u] %s' % (idx + 1, tot_rows, run_name))
        replaces = {}
        for k, v in zip(header, row):
            if v.isdecimal():
                d = float(v)
                v = v + '.f' if math.floor(d) == d else v + 'f'
            replaces[k] = v
        each_do(idx, run_name, replaces)


if __name__ == '__main__':
    output = tsv_product('auto_eval.tsv')
    with open('product.tsv', 'w') as fh:
        fh.write(output + '\n')
    header, rows = tsv_eval_read('product.tsv')
    def replace_and_remake(idx, run_name, replaces):
        replace_source_code('./template', replaces)
        remake('../')
    tsv_eval_do(header, rows, replace_and_remake)
