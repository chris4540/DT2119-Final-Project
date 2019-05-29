"""
A draft of plotting the acc. vs % labeled data at different temperature
"""
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("test_temp.csv", index_col=0)
    df.index.name = "% labeled samples"
    df.columns.name = "Temp"
    for column in df:
        df[column] = df[column].str.rstrip('%').astype('float') / 100.0

    ax = df.plot(kind='line', style='x-')
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.68, 0.73])
    fig = ax.get_figure()
    fig.savefig("semi_plot.png", bbox_inches='tight')