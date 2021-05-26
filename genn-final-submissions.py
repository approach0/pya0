#!/usr/bin/python3
import subprocess
import os
from pya0.eval import evaluate_run


def shell(cmd):
    print('[shell]', cmd)
    r = os.system(cmd)
    if r != 0:
        quit(1)


def dele_2(fname):
    shell('cat %s | awk \'{print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}\' > %s.swap' % (fname, fname))
    shell(f'mv {fname}.swap {fname}')


def swap_2_3(fname):
    shell('cat %s | awk \'{print $1 "\t" $3 "\t" $2 "\t" $4 "\t" $5 "\t" $6}\' > %s.swap' % (fname, fname))
    shell(f'mv {fname}.swap {fname}')


def preprocess_anserini_task2(tmprun):
    shell(f'sed -i -e "s/^/B./g" {tmprun}')
    shell(f'sed -i -e "s/-/ /g" {tmprun}')
    shell(f'sed -i -e "s/Q0//g" {tmprun}')


def gen_final_runs(only_anserini=False):
    shell('mkdir -p tmp')
    shell('mkdir -p runs/2020')
    shell('mkdir -p runs/2021')

    with open('final-run-generator.tsv', 'r') as fh:
        for ln, line in enumerate(fh):
            fields = line.split('\t')
            fields = [f.strip() for f in fields]
            if ln == 0:
                keys = fields
                continue # skip header
            elif line.startswith('#'):
                continue # skip commented row
            r = dict(list(zip(keys, fields)))
            # output path/name
            name = '+'.join([f.replace(' ', '_').replace('/', '_') for f in fields])
            output=f'./runs/{r["year"]}/{name}.run'
            a0save=f'./runs/{r["year"]}/a0-{name}.run'
            if only_anserini:
                shell(f'cp {a0save} a0.run')
            else:
                # use which a0 parameters?
                shell(f'cp auto_eval-{r["a0_param"]}.tsv auto_eval.tsv')
                # run math search
                if r["year"] == "2021":
                    collection = f'arqmath-{r["year"]}-{r["task"]}-refined'
                else:
                    collection = f'arqmath-{r["year"]}-{r["task"]}'
                cmd = f'python3 -m pya0 --index index-{r["task"]}-2021 --collection {collection} --auto-eval tmp' # + ' --select-topic B.202'
                run_args = cmd.split()
                for k in ['a0_math_exp', 'a0_rm3']:
                    if r[k] != '':
                        run_args += r[k].split()
                shell('rm -f tmp/*') # clean up a0 runfile output directory
                print(run_args)
                subprocess.run(run_args)
                shell(f'mv tmp/*.run a0.run') # take output from last stage as input of this stage
                shell(f'cp a0.run {a0save}') # save a0 run
            # preprocessing for approach0
            if r["task"] == 'task2':
                swap_2_3('a0.run')
            # run 2nd-stage
            _2nd_stage = r["2nd_stage"]
            ans_save=f'./runs/{r["year"]}/anserini-{name}.run'
            if _2nd_stage != '':
                # preprocessing for anserini
                shell(f'cp {r["anserini_run"]} anserini.run')
                if r["task"] == 'task1':
                    shell(f'sed -i -e "s/^/A./g" anserini.run')
                    shell(f'cp anserini.run {ans_save}')
                    dele_2(ans_save)
                elif r["task"] == 'task2':
                    preprocess_anserini_task2('anserini.run')
                    shell(f'cp anserini.run {ans_save}')
                    swap_2_3('anserini.run')
                # invoke 2nd-stage
                shell('rm -f ./mergerun-*')
                cmd = f'python3 -m pya0 {_2nd_stage}'
                run_args = cmd.split()
                print(run_args)
                subprocess.run(run_args)
                shell(f'mv mergerun-*.run {output}')
            else:
                shell(f'cp a0.run {output}')
            # post-processing
            if r["task"] == 'task1':
                dele_2(output)
            elif r["task"] == 'task2':
                swap_2_3(output)
            shell(f'rm -f a0.run anserini.run')
            shell(f'sed -i -e "s/ /\\t/g" {output}')


def gen_tsv_from_2020():
    header = None
    rows = []
    with open('final-run-generator.tsv', 'r') as fh:
        for ln, line in enumerate(fh):
            fields = line.split('\t')
            fields = [f.strip() for f in fields]
            if ln == 0:
                keys = fields
                continue # skip header
            elif line.startswith('#'):
                continue # skip commented row
            r = dict(list(zip(keys, fields)))
            if r['year'] == '2020':
                name = '+'.join([f.replace(' ', '_').replace('/', '_') for f in fields])
                path = f'./runs/{r["year"]}/{name}.run'
                if r["year"] == "2021":
                    collection = f'arqmath-{r["year"]}-{r["task"]}-refined'
                else:
                    collection = f'arqmath-{r["year"]}-{r["task"]}'
                header, row = evaluate_run(collection, path)
                rows.append(fields + row)
            else:
                rows.append(fields)
            print(rows[-1])
    if header:
        with open('final-run-results.tsv', 'w') as fh:
            print('\t'.join(keys + header), file=fh)
            for row in rows:
                print('\t'.join(row), file=fh)


def gen_submissions(root):
    import uuid
    folder = uuid.uuid4()
    with open('final-run-generator.tsv', 'r') as fh:
        for ln, line in enumerate(fh):
            fields = line.split('\t')
            fields = [f.strip() for f in fields]
            if ln == 0:
                continue # skip header
            elif line.startswith('#'):
                continue # skip commented row
            name = '+'.join([f.replace(' ', '_').replace('/', '_') for f in fields])
            if name.find('approach0') >= 0:
                year = name.split('+')[0]
                task = name.split('+')[1]
                final_name = name.split('+')[-1]
                task = 'Task1-QA' if task == 'task1' else 'Task2-Formulas'
                src_path = f'./runs/{year}/{name}.run'
                dst_dir = f'{root}/{folder}/{task}/{year}'
                shell(f'mkdir -p {dst_dir}')
                shell(f'cp {src_path} \t {dst_dir}/{final_name}.tsv')
    shell(f'cd {root}/{folder}')


if __name__ == '__main__':
    gen_final_runs(only_anserini=False)
    #gen_tsv_from_2020()
    gen_submissions('/tuna1/scratch/w32zhong/arqmath/2021-submission')
    #preprocess_anserini_task2('new.run')
    #dele_2("./a0-textonly.run")
