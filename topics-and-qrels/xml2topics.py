from xmlr import xmliter

input_xml='topics.arqmath-2021-task2.origin.xml'
output_txt='topics.arqmath-2021-task2.txt'

with open(output_txt, 'w') as fh:
    for attrs in xmliter(input_xml, 'Topic'):
        qid = attrs['@number']
        latex = attrs['Latex']
        print(f'{qid}\t{latex}', file=fh)
