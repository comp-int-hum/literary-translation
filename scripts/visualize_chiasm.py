# we take in a .jsonl file where each line has a field n (for size of chiasm), i (for start position of the embeddings) and p (for p-value)
# we want to produce a visualization of 1 horizontal block for every verse (~7k) and if there is a statistically significant chiasmus of size n, color that block (under some threshold thres)
# we want to color the block with a color that is a function of the p-value

import argparse
import numpy as np
import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Location

# we want to color the block with a color that is a function of the p-value
def get_color(p, thres):
    if p < thres:
        return "red"
    else:
        return "white"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", dest="scores", help="Scores file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--thres", dest="thres", type=float, help="Threshold for significance")
    args = parser.parse_args()

    with open(args.scores, "rt") as ifd:
        data = [json.loads(line) for line in ifd]
    print(data[:5])
    # make this a dataframe
    df = pd.DataFrame.from_records(data)
    print(df.head())
    # keep the rows where the p-value is less than the threshold
    # df = df[df["p"] < args.thres]
    # print(df.head())
    # let's take there n == 2
    # df = df[df["n"] == 10]
    # and plot like a coloured bar code where p < thres is red and p > thres is white
    df["color"] = df["p"].apply(lambda x: 1 if x < args.thres else 0)
    # multiply it by n
    df["color"] *= df["n"]
    print(df['color'].value_counts())
    # want to groupby i, keep the n with the smallest p-value
    df = df.groupby("i").apply(lambda x: x.loc[x["p"].idxmin()]).reset_index(drop=True)
    print(df.head())
    print(df['color'].value_counts())
    exit()
    # print(df.head(20))
    # make a plot
    # Define the colormap for values 0-5
    # make the fig very wide and not tall
    plt.figure(figsize=(20,1))
    # let's make our own color map, there 0 is white, 1 is blue
    # cmap = sns.color_palette(["white", "blue"])
    plt.imshow(df["color"].values.reshape(1, -1), cmap="tab20", aspect="auto")
    # Add a colorbar
    # don't need y axis
    plt.axis("off")
    plt.colorbar()
    plt.xlim(0,400)
    # Show the plot
    plt.show()
    
    exit()
    fig, ax = plt.subplots()
    # bar plot?
    # just plot the color column
    plt.plot(df["color"].values)
    # sns.barplot(x="i", y="color", data=df, ax=ax)
    # sns.heatmap(df["color"].values.reshape(1,-1), ax=ax, cmap="RdYlBu", cbar=True)
    # plt.xlim(0,400)
    # want the xticks to be increments of 1000
    # ax.set_xticks(np.arange(0, len(df), 1000))
    plt.savefig(args.output)
    # group by i, make the p_values into a list and n values in a list
    # df = df.groupby("i")[["n", "p"]].apply(lambda x: x.to_dict(orient='records')).reset_index()
    # print(df.head())
    
    # df = df.groupby("i")["p"].apply(list).reset_index()

    # print(df.head())
    # we want to produce a visualization of 1 horizontal block for every verse (~7k) and if there is a statistically significant chiasmus of size n, color that block (under some threshold thres)
    # 