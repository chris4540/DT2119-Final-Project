import pandas as pd

def parse_percentage(df):
    for column in df:
        df[column] = df[column].str.rstrip('%').astype('float') / 100.0
    return df

if __name__ == "__main__":
    test_acc = parse_percentage(pd.read_csv("cmp_test_acc.csv"))
    valid_acc = parse_percentage(pd.read_csv("cmp_valid_acc.csv"))
    train_acc = parse_percentage(pd.read_csv("cmp_train_acc.csv"))

    # add prefix
    test_acc = test_acc.add_prefix('Test Acc. ')
    valid_acc = valid_acc.add_prefix('Valid Acc. ')
    train_acc = train_acc.add_prefix('Train Acc. ')

    big_df = pd.concat([train_acc, valid_acc, test_acc], axis=1)
    linestyles = []
    colors = []
    styles = dict()
    for c in big_df.columns:
        if "adam" in c:
            color = "xkcd:salmon"
        elif "stepLR" in c:
            color = 'xkcd:mint green'
        elif "cyclicLR" in c:
            color = "xkcd:blue purple"
        else:
            color = ""
        # ----------------------------
        if "Train" in c:
            line = '-'
        elif "Valid" in c:
            line = "--"
        elif "Test" in c:
            line = ":"
        else:
            line = ""

        styles[c] = line
        linestyles.append(line)
        colors.append(color)


    # plot
    ax = big_df.plot(kind='line', color=colors, style=styles)
    fig = ax.get_figure()
    fig.savefig("cmp_optim.png", bbox_inches='tight', dpi=800)