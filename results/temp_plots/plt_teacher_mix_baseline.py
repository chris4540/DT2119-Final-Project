import pandas as pd

def parse_percentage(df):
    for column in df:
        df[column] = df[column].str.rstrip('%').astype('float') / 100.0
    return df
if __name__ == "__main__":
    # make the teacher-student model with highest temp first
    t_s_valid_df = pd.read_csv("valid_temp.csv", index_col=0)
    t_s_test_df = pd.read_csv("test_temp.csv", index_col=0)

    # parse percentage
    t_s_valid_df = parse_percentage(t_s_valid_df)
    t_s_test_df = parse_percentage(t_s_test_df)

    # obtain the max temp for different partition of labeled data
    max_temps = t_s_valid_df.idxmax(axis=1)

    # make a new dataframe
    df = pd.DataFrame(max_temps, columns=['max_temp'])
    df['valid_acc'] = t_s_valid_df.lookup(*zip(*max_temps.items()))
    df['test_acc'] = t_s_test_df.lookup(*zip(*max_temps.items()))
    # re ordering
    df = df[['valid_acc', 'test_acc', 'max_temp']]

    # save it
    df.to_csv("max_temp_valid_test_acc.csv")
    # ====================================================================
    # load baseline csv
    baseline_acc = pd.read_csv("baseline_acc.csv", index_col=0)
    baseline_acc = parse_percentage(baseline_acc)
    baseline_acc = baseline_acc.add_prefix("baseline_")

    # ====================================================================
    # load teacher csv
    teacher_acc = pd.read_csv("teacher_acc.csv", index_col=0)
    teacher_acc = parse_percentage(teacher_acc)
    teacher_acc = teacher_acc.add_prefix("teacher_")

    # teacher-student pair
    t_s_acc = df.drop("max_temp", axis=1)
    t_s_acc = t_s_acc.add_prefix('teacher_student_')
    # ==============================================================
    big_df = pd.concat([t_s_acc, teacher_acc, baseline_acc], axis=1)
    # build style dict
    styles = dict()
    for col in big_df:
        if 'valid' in col:
            styles[col] = 'x--'
        else:
            styles[col] = 'x-'

    ax = big_df.plot(kind='line', style=styles)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("% labeled samples")
    ax.set_xlim([0, None])
    fig = ax.get_figure()
    fig.savefig("cmp_acc.png", bbox_inches='tight', dpi=800)