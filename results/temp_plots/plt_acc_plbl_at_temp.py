import pandas as pd

def parse_percentage(df):
    for column in df:
        df[column] = df[column].str.rstrip('%').astype('float') / 100.0
    return df

if __name__ == "__main__":
    test_acc_at_temp = parse_percentage(pd.read_csv("test_temp.csv", index_col=0))
    valid_acc_at_temp = parse_percentage(pd.read_csv("valid_temp.csv", index_col=0))

    # rename columns
    valid_acc_at_temp.columns = ["Valid Acc. at T = %s" % c for c in valid_acc_at_temp.columns]
    test_acc_at_temp.columns = ["Test Acc. at T = %s" % c for c in test_acc_at_temp.columns]

    big_df = pd.concat([valid_acc_at_temp, test_acc_at_temp], axis=1)
    # build style dict
    styles = dict()
    for col in big_df:
        if 'Valid Acc.' in col:
            styles[col] = 'x--'
        else:
            styles[col] = 'x-'
    ax = big_df.plot(kind='line', style=styles)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("% labeled samples")
    ax.set_xlim([0, None])
    fig = ax.get_figure()
    fig.set_size_inches(8, 6)
    fig.savefig("temp_plot.png", bbox_inches='tight', dpi=800)