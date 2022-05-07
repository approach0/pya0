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
        table.append([])
        for run in files:
            qid = f'A.{300 + row}'
            run_dict = all_run_dict[run]
            if qid in run_dict:
                content = run_dict[qid][0]['content']
            else:
                content = 'KeyError: ' + qid
            table[row].append(content)
    print(table)
    quit()
        

fire.Fire(visualize)
