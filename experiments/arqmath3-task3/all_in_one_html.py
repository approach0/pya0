import fire
import sys
sys.path.insert(0, '.')
from pya0.mergerun import parse_task3_file

def visualize(*files):
    table = [['.']]
    all_run_dict = {}
    for run in files:
        table[0].append(run)
        run_dict, _ = parse_task3_file(run)
        all_run_dict[run] = run_dict

    for row in range(1, 101):
        qid = f'A.{300 + row}'
        table.append([qid])
        for run in files:
            run_dict = all_run_dict[run]
            if qid in run_dict:
                content = run_dict[qid][0]['content']
            else:
                content = 'KeyError: ' + qid
            table[row].append(content)

    with open(f'visualization/all_in_one.html', 'w') as fh:
        fh.write('<html>\n')
        # head
        fh.write('<head>\n')
        fh.write('<style>\n')
        fh.write('table, th, td { border: 1px solid black; }\n')
        fh.write('</style>\n')
        fh.write('</head>\n')
        # table
        fh.write('<table>\n')
        #for row in range(0, 2): ### DEBUG
        for row in range(0, 101):
            print(row)
            fh.write('<tr>\n')
            for cell in table[row]:
                fh.write('<td>\n')
                fh.write(cell)
                fh.write('</td>\n')
            fh.write('</tr>\n')
        fh.write('</table>\n')

        # mathJax
        mathjax_cdn = "https://cdn.jsdelivr.net/npm/mathjax@3.2.0/es5/tex-chtml-full.js"
        fh.write('<script> window.MathJax ={' +
            "loader: { source: {'[tex]/AMScd': '[tex]/amscd'} }," +
            'tex: { inlineMath: [ ["$","$"] ] }' +
        '}; </script>')
        fh.write(f'<script type="text/javascript" src="{mathjax_cdn}"></script>')
        # end document
        fh.write('</body></html>\n')

fire.Fire(visualize)
