import fire
import pandas as pd
import matplotlib.pyplot as plt


def read_trec_output_by_query(file_path):
    x, y = [], []
    with open(file_path) as fh:
        for line in fh:
            line = line.rstrip()
            fields = line.split()
            x.append(fields[1])
            y.append(float(fields[2]))
    return x, y


def compare_by_query(file1, file2):
    labels, y1 = read_trec_output_by_query(file1)
    _, y2 = read_trec_output_by_query(file2)
    if False:
        a = set(_)
        b = set(labels)
        print(a.difference(b))
        print(b.difference(a))
    df = pd.DataFrame(y1, columns=['y1'], index=labels)
    df['y2'] = pd.DataFrame(y2, index=_)
    print(df)
    df.plot(kind='bar')
    plt.gca().xaxis.set_tick_params(rotation=90)
    plt.ylabel('NDCG')


if __name__ == '__main__':
    fire.Fire(compare_by_query)
