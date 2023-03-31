import os
import re
import fire
import pandas as pd


def plot(file_path):
    m = re.search(r'math_pruner_updates--(\S+).log$', file_path)
    topic = 'Unknown' if m is None else m.group(1)

    df = pd.read_csv(file_path, header=None, sep="\s+",
            names=['n_qnodes', 'g_th', 'pivot', 'n_iters']
        )

    x = list(range(len(df)))
    z1 = list(df['n_qnodes'])
    z2 = list(df['g_th'])
    z3 = list(df['pivot'])
    z4 = list(df['n_iters'])

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    color1, color2, color3, color4 = plt.cm.viridis([0, .3, .6, .9])
    p1 = ax3.plot(x, z1, marker=',',label="query OPT nodes", color=color1)
    p2 = ax.plot(x, z3, marker=',',label="requirement set", color=color2)
    p3 = ax.plot(x, z4, marker=',',label="inverted lists", color=color3)
    p4 = ax2.plot(x, z2, marker=',',label="threshold", color=color4)

    ax.yaxis.label.set_color(p2[0].get_color())
    ax2.yaxis.label.set_color(p4[0].get_color())
    ax3.yaxis.label.set_color(p1[0].get_color())

    ax.yaxis.get_major_locator().set_params(integer=True)
    ax3.yaxis.get_major_locator().set_params(integer=True)

    ax.legend(handles=p1+p2+p3+p4, loc='upper left')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Number')
    ax2.set_ylabel(r'Threshold ($\theta$)')
    ax3.set_ylabel(r'OPT nodes')

    plt.title(topic)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'math_pruner_updates_plot--{topic}.eps')


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(plot)
