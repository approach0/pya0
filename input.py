import subprocess
task = 'task2'
cmd = f'python3 -m pya0 --index index-{task}-2021 --collection arqmath-2021-{task} --trec-output adhoc.run --visualize-run adhoc.run --topk 20 --query'.split()
with open('input.tsv', 'r') as fh:
    line = fh.read().rstrip()
    fields = line.split('\t')
    formulas = [f'${f}$' for f in fields[1:]]
    keywords = filter(lambda x: len(x.strip()) > 0, [fields[0]] + formulas)
    keywords = list(keywords)
    subprocess.run(cmd + keywords)
