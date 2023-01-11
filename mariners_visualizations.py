import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def explore_relationships(df):
    df.replace(to_replace=["L", "R"], value=[1, 0], inplace=True)
    stats = list(df.columns)
    excluded = ["play_id", "game_id", "catcher_id", "runner_id", "pitcher_id", "batter_id", "is_cs"]
    for col in stats:
        if col not in excluded:
            user_input = input("Display next graph? (Y/N): ")
            if user_input == "Y":
                plt.scatter(df[col], df["is_cs"])
                plt.title("Caught stealing vs. {}".format(col))
                plt.show()
            else:
                break



def ind_vs_group_scatter(df, indiv_id, statistic, id_col="catcher_id"):
    indiv_stats = df.loc[df[id_col] == indiv_id][statistic]
    group_stats = df[statistic]

    ind_realizations = len(indiv_stats)
    group_realizations = len(group_stats)

    group_hist, bins = np.histogram(group_stats)
    indiv_hist, _ = np.histogram(indiv_stats, bins=bins)
    # generates labels for bar chart from histogram bins
    labels = ["{0:.2f}-{1:.2f}".format(bins[i], bins[i + 1]) for i in range(0, len(bins) - 1)]
    axis_points = np.arange(len(labels))
    plt.bar(axis_points-0.2, group_hist/group_realizations, 0.4, label="group distribution")
    plt.bar(axis_points+0.2, indiv_hist/ind_realizations, 0.4, label="individual results")
    plt.xticks(axis_points, labels)
    plt.title("Distribution of {} for player and group".format(statistic))
    plt.xlabel("Bins for {}".format(statistic))
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def scroll_charts(df, indiv_id, id_col="catcher_id"):
    stats = list(df.columns)
    excluded = ["play_id", "game_id", "catcher_id", "runner_id", "pitcher_id", "batter_id", "pitcher_throws",
                "batter_side"]
    for col in stats:
        if col not in excluded:
            user_input = input("Display next graph? (Y/N): ")
            if user_input == "Y":
                ind_vs_group_scatter(df, indiv_id, col, id_col)
            else:
                break


if __name__ == "__main__":
    data = pd.read_csv("mariners_train.csv").dropna()
    #explore_relationships(data)
    scroll_charts(data, "203ed108", "catcher_id")
