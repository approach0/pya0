import pandas as pd
import os
import csv
import pdb
df = pd.read_csv("experiments/arqmath3-task3/fix_sentences.tsv", sep='\t', header=None, dtype=str)
for index, row in df.iterrows():
    target_file, qid, repl = list(row)
    qid = qid.strip()
    if repl.startswith('runs'):
        df_repl = pd.read_csv(repl, sep='\t', header=None, dtype=str)
        content = df_repl[df_repl.iloc[:,0] == qid]
        content = content.iloc[0,5]
    else:
        content = repl
    content = content.strip()
    print('target:', target_file, qid)
    df_target = pd.read_csv(target_file, sep='\t', header=None, dtype=str)
    with open('tmp.tsv', 'w') as fh:
        for _, row_target in df_target.iterrows():
            fields = list(row_target)
            fields = list(map(str, fields))
            if list(row_target)[0].strip() == qid:
                print(qid, '-->', content[:16], '...')
                fields[-1] = content
            fields[-1] = '"' + fields[-1] + '"'
            fh.write('\t'.join(fields))
            fh.write('\n')
    os.replace('tmp.tsv', target_file)
