import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.ticker as plticker

def parse_percentage(df):
    for column in df:
        df[column] = df[column].str.rstrip('%').astype('float') / 100.0
    return df

if __name__ == "__main__":
    test_acc_at_temp = parse_percentage(pd.read_csv("test_temp.csv", index_col=0))
    valid_acc_at_temp = parse_percentage(pd.read_csv("valid_temp.csv", index_col=0))

    # assign temp with color
    temps = list(valid_acc_at_temp.columns)
    # rename columns
    valid_acc_at_temp.columns = ["Valid Acc. at T = %s" % c for c in valid_acc_at_temp.columns]
    test_acc_at_temp.columns = ["Test Acc. at T = %s" % c for c in test_acc_at_temp.columns]

    idx = np.linspace(0, 1, 9)
    temp2color = {
        '0.5': cm.Set1(idx[8]),
        '1': cm.Set1(idx[1]),
        '2': cm.Set1(idx[2]),
        '4': cm.Set1(idx[3]),
        '6': cm.Set1(idx[4]),
        '8': cm.Set1(idx[5]),
        '10': cm.Set1(idx[6]),
        '20': cm.Set1(idx[7]),
        '50': cm.Set1(idx[0]),
        '100': 'xkcd:dusty blue',
        '200': 'xkcd:neon blue',
        '500': 'xkcd:cobalt blue',
        '800': 'xkcd:bright purple'
    }

    big_df = pd.concat([valid_acc_at_temp, test_acc_at_temp], axis=1)
    # build style dict
    styles = dict()
    colors = list()
    for col in big_df.columns:
        if 'Valid Acc.' in col:
            styles[col] = '--'
        else:
            styles[col] = '-'
        # select color
        color = temp2color[col.split("= ")[-1]]
        colors.append(color)

    ax = big_df.plot(kind='line', style=styles, color=colors, marker='x')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("% labeled samples")
    ax.set_xlim([0, None])
    loc = plticker.MultipleLocator(base=.01) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    fig = ax.get_figure()
    fig.set_size_inches(12, 9)
    fig.savefig("temp_plot.png", bbox_inches='tight', dpi=400)
