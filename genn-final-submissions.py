#!/usr/bin/python3
import subprocess
import os

def shell(cmd):
    print('[shell]', cmd)
    r = os.system(cmd)
    if r != 0:
        quit(1)

shell('mkdir -p tmp')
shell('rm -f tmp/*')
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
        # run math search
        cmd = f'python3 -m pya0 --index index-{r["task"]}-{r["year"]} --collection arqmath-{r["year"]}-{r["task"]} --auto-eval tmp' # + ' --select-topic B.202'
        run_args = cmd.split()
        for k in ['a0_math_exp', 'a0_rm3']:
            if r[k] != '':
                run_args += r[k].split()
        print(run_args)
        subprocess.run(run_args)
        # output path/name
        name = '+'.join([f.replace(' ', '_').replace('/', '_') for f in fields])
        output=f'./runs/{r["year"]}/{name}.run'
        # run 2nd-stage
        shell(f'mv tmp/*.run a0.run')
        shell(f'cp {r["anserini_run"]} anserini.run')
        shell(f'rm -f ./merged-* ./concate-*')
        _2nd_stage = r["2nd_stage"]
        if _2nd_stage != '':
            cmd = f'python3 -m pya0 {_2nd_stage}'
            run_args = cmd.split()
            print(run_args)
            subprocess.run(run_args)
            if _2nd_stage.find('concate') >= 0:
                shell(f'mv concate-*.run {output}')
            else:
                shell(f'mv merged-*.run {output}')
        else:
            shell(f'mv a0.run {output}')
        shell(f'sed -i -e "s/ /\\t/g" {output}')
