#!/usr/bin/python3
import subprocess
import os
from pya0.eval import evaluate_run


def shell(cmd):
    print('[shell]', cmd)
    r = os.system(cmd)
    if r != 0:
        quit(1)


def swap_2_3(fname):
    shell('cat %s | awk \'{print $1 "\t" $3 "\t" $2 "\t" $4 "\t" $5 "\t" $6}\' > %s.swap' % (fname, fname))
    shell(f'mv {fname}.swap {fname}')


def gen_final_runs():
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
            # use which a0 parameters?
            shell(f'cp auto_eval-{r["a0_param"]}.tsv auto_eval.tsv')
            # run math search
            collection = f'arqmath-{r["year"]}-{r["task"]}'
            cmd = f'python3 -m pya0 --index index-{r["task"]}-2021 --collection {collection} --auto-eval tmp' # + ' --select-topic B.202'
            run_args = cmd.split()
            for k in ['a0_math_exp', 'a0_rm3']:
                if r[k] != '':
                    run_args += r[k].split()
            shell('rm -f tmp/*') # clean up a0 runfile output directory
            print(run_args)
            subprocess.run(run_args)
            # output path/name
            name = '+'.join([f.replace(' ', '_').replace('/', '_') for f in fields])
            output=f'./runs/{r["year"]}/{name}.run'
            # run 2nd-stage
            shell(f'mv tmp/*.run a0.run') # take output from last stage as input of this stage
            _2nd_stage = r["2nd_stage"]
            if _2nd_stage != '':
                # preprocessing
                shell(f'cp {r["anserini_run"]} anserini.run')
                if r["task"] == 'task1':
                    shell(f'sed -i -e "s/^/A./g" anserini.run')
                elif r["task"] == 'task2':
                    shell(f'sed -i -e "s/^/B./g" anserini.run')
                    shell(f'sed -i -e "s/-/ /g" anserini.run')
                    shell(f'sed -i -e "s/Q0//g" anserini.run')
                    swap_2_3('a0.run')
                    swap_2_3('anserini.run')
                # invoke 2nd-stage
                shell('rm -f ./mergerun-*')
                cmd = f'python3 -m pya0 {_2nd_stage}'
                run_args = cmd.split()
                print(run_args)
                subprocess.run(run_args)
                shell(f'mv mergerun-*.run {output}')
                # post-processing
                if r["task"] == 'task2':
                    swap_2_3(output)
            else:
                shell(f'cp a0.run {output}')
            shell(f'sed -i -e "s/ /\\t/g" {output}')


def evaluate_from_2020():
    # Get a sense of how good each search model is from 2020 runs
    for folder, _, files in os.walk('runs/2020'):
        for filename in files:
            path = os.path.join(folder, filename)
            for task in ['task1', 'task2']:
                if filename.find(task) >= 0:
                    shell(f'./eval-arqmath-{task}.sh {path} | grep all')


def gen_tsv_from_2020():
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
                collection = f'arqmath-{r["year"]}-{r["task"]}'
                header, row = evaluate_run(collection, path)
                rows.append(fields + row)
            else:
                rows.append(fields)
            print(rows[-1])
    with open('final-run-results.tsv', 'w') as fh:
        print('\t'.join(keys + header), file=fh)
        for row in rows:
            print('\t'.join(row), file=fh)


def gen_submissions():
    shell('mkdir -p runs/2020/submission')
    shell('mkdir -p runs/2021/submission')
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
                final_name = name.split('+')[-1]
                src_path = f'./runs/{year}/{name}.run'
                dst_path = f'./runs/{year}/submission/{final_name}.tsv'
                shell(f'cp {src_path} \t {dst_path}')
    shell('ls runs/*/submission')


if __name__ == '__main__':
    gen_final_runs()
    #evaluate_from_2020()
    #gen_tsv_from_2020()
    gen_submissions()
