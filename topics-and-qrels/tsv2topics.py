from xmlr import xmliter

input_xml='topics.arqmath-2021-task2-refined.origin.tsv'
output_txt='topics.arqmath-2021-task2-refined.txt'

with open(output_txt, 'w') as fh:
    with open(input_xml, 'r') as fin:
        for line in fin:
            line = line.rstrip()
            fields = line.split('\t')
            qid = fields[0]
            if len(fields) > 2:
                formulas = fields[2:]
            else:
                formulas = fields[1:]
            fh.write(f'{qid}\t' + '\t'.join(formulas) + '\n')
