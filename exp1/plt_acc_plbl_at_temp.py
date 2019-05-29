"""
A draft of plotting the acc. vs % labeled data at different temperature
"""
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("result.csv", index_col=0)
    df.index.name = "% labeled samples"
    df.columns.name = "Temp"
    print(df)
    ax = df.plot(kind='line', style='x-')
    ax.set_ylabel("Accuracy")
    fig = ax.get_figure()
    fig.savefig("semi_plot.png", bbox_inches='tight')