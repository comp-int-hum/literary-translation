import argparse
import numpy as np
import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utils import Location
import os
import glob
import multiprocessing as mp

# sourced from: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", dest="work", help="Work directory")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--thres", dest="thres", type=float, help="Threshold for significance")
    args = parser.parse_args()

    score_files = glob.glob(os.path.join(args.work, "**/*-chiasm.json.gz"), recursive=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    OUTPUT = args.output


    def plot_scores(score_file):
        with gzip.open(score_file, "rb") as ifd:
            data = [json.loads(line) for line in ifd]
        
        df = pd.DataFrame.from_records(data)
        # N is the number of unique n values in the df
        df["color"] = df["p"].apply(lambda x: 1 if x < args.thres else 0)
        df["color"] *= df["n"]
        # group by i and keep the n with the smallest p-value
        df = df.groupby("i").apply(lambda x: x.loc[x["p"].idxmin()]).reset_index(drop=True)
        N = len(df["color"].unique())

        # now we save the plot 
        plt.figure(figsize=(40,5))
        plt.imshow(df["color"].values.reshape(1, -1), cmap=discrete_cmap(N, 'PuBu'), aspect="auto", norm=mpl.colors.BoundaryNorm(sorted(df['color'].unique()), N))
        # want the ticks of the colorbar to be the n values
        plt.colorbar(ticks=sorted(df['color'].unique()))
        plt.axis("off")
        # the path of the file looks like this "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-chiasm.json.gz"
        title = " ".join(score_file.split("/")[1:]).replace("-chiasm.json.gz", "")
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT, os.path.basename(score_file).replace(".json.gz", ".png")), bbox_inches='tight')

    # now we want to map this function to all of the score files with pool
    plot_scores(score_files[0])
    # with mp.Pool() as pool:
        # pool.map(plot_scores, score_files)
    # exit()



    # with gzip.open(args.scores, "rb") as ifd:
    #     data = [json.loads(line) for line in ifd]
  
    # # make this a dataframe
    # df = pd.DataFrame.from_records(data)
 
    # df["color"] = df["p"].apply(lambda x: 1 if x < args.thres else 0)
    # # multiply it by n
    # df["color"] *= df["n"]

    # want to groupby i, keep the n with the smallest p-value
    
    
    # fig = plt.figure(figsize=(40,5))
    
    # N=args.n
    # plt.imshow(df["color"].values.reshape(1, -1), cmap=discrete_cmap(N, 'PuBu'), aspect="auto")
    # plt.colorbar(ticks=range(N))
    # plt.clim(-0.5, N - 0.5)
    # # Add a colorbar
    # # don't need y axis
    # plt.axis("off")
    # # plt.colorbar()
    # # plt.xlim(0,400)
    # # Show the plot
    # plt.show()
    
    # exit()
    # fig, ax = plt.subplots()
    # # bar plot?
    # # just plot the color column
    # plt.plot(df["color"].values)
    # # sns.barplot(x="i", y="color", data=df, ax=ax)
    # # sns.heatmap(df["color"].values.reshape(1,-1), ax=ax, cmap="RdYlBu", cbar=True)
    # # plt.xlim(0,400)
    # # want the xticks to be increments of 1000
    # # ax.set_xticks(np.arange(0, len(df), 1000))
    # plt.savefig(args.output)
    # # group by i, make the p_values into a list and n values in a list
    # # df = df.groupby("i")[["n", "p"]].apply(lambda x: x.to_dict(orient='records')).reset_index()
    # # print(df.head())
    
    # # df = df.groupby("i")["p"].apply(list).reset_index()

    # # print(df.head())
    # # we want to produce a visualization of 1 horizontal block for every verse (~7k) and if there is a statistically significant chiasmus of size n, color that block (under some threshold thres)
    # # 