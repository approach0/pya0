import os
import fire
import pandas


def filter_topics(topic_info_csv, run_file, column_name, filter_value):
    df = pandas.read_csv(topic_info_csv)
    df = df[df[column_name] == filter_value]
    filter_topics = set(df['Topic'].to_list())
    print(f'{run_file}: Filtered by {column_name}={filter_value}', end=': ')
    print(filter_topics)
    output_filename = f'{run_file}-{column_name}-{filter_value}'
    cnt = 0
    with open(run_file, 'r') as fh, open(output_filename, 'w') as w_fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            topic = fields[0]
            if topic in filter_topics:
                print(line, file=w_fh)
                cnt += 1
    if cnt == 0:
        print('Warning: no leftout result.')
    else:
        print(f'{cnt} result(s).')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(filter_topics)
