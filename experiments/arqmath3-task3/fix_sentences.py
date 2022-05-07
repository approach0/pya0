import pandas as pd
import os
import csv
import pdb
df = pd.read_csv("experiments/arqmath3-task3/fix_sentences.tsv", sep='\t', header=None, dtype=str)
for index, row in df.iterrows():
    target_file, qid, repl = list(row)
    target_file = target_file.strip()
    qid = qid.strip()
    repl = repl.strip()
    print('source:', repl, qid)
    print('target:', target_file, qid)
    if target_file == repl:
        print('skip...')
        continue
    if repl.startswith('runs'):
        df_repl = pd.read_csv(repl, sep='\t', header=None, dtype=str)
        content = df_repl[df_repl.iloc[:,0] == qid]
        content = content.iloc[0,5]
    else:
        content = repl
    content = content.strip()
    df_target = pd.read_csv(target_file, sep='\t', header=None, dtype=str)
    with open('tmp.tsv', 'w') as fh:
        last = 300
        for _, row_target in df_target.iterrows():
            fields = list(row_target)
            fields = list(map(str, fields))
            curr = int(list(row_target)[0].split('.')[1].strip())
            insert = False
            #print('A.' + str(last + 1), qid)
            if 'A.' + str(last + 1) == qid:
                if curr == last + 1:
                    print(qid, '-->', content[:32], '...')
                    fields[-1] = content
                else:
                    copy_fields = fields[:]
                    copy_fields[-1] = content
                    print(qid, '==NEW==>', content[:32], '...')
                    copy_fields[0] = qid
                    copy_fields[-1] = '"' + copy_fields[-1] + '"'
                    insert = True
                    fh.write('\t'.join(copy_fields))
                    fh.write('\n')
                    #print('\t'.join(copy_fields))
                    last = last + 1
            last = last + 1
            fields[-1] = '"' + fields[-1] + '"'
            fh.write('\t'.join(fields))
            fh.write('\n')
            #if insert:
            #    print('\t'.join(fields))
            #    print(last)
    os.replace('tmp.tsv', target_file)
