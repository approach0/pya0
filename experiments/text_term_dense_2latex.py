import re
import pandas as pd
pd.set_option('max_colwidth', None)

df = pd.read_csv('text_term_dense.report', delimiter='\s+')
df = df.drop(['BPref', 'Judge'], axis='columns')

for i, row in df.iterrows():
    system = row['System']
    system = system.replace('_', '.')
    match = re.search(r'mergerun--([\d.]+)W.arqmath3-cocomae.run--([\d.]+)W.arqmath3-tex.run--([\d.]+)W.arqmath3-term.run', system)
    w1, w2, w3 = match.group(1), match.group(2), match.group(3)
    w1, w2, w3 = float(w1), float(w2), float(w3)
    #print(system)
    #print(w1, w2, w3)
    df.at[i, 'System'] = f'{w1:.2f} / {w2:.2f} / {w3:.2f}'

latex = df.to_latex(index=True, multirow=True, escape=False, column_format='lr|ccccc')
print(latex)
