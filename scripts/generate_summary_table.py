# we want to load in a bunch of score files, and make a table of what percentage of verses have a statistically significant chiasmus

import argparse as ap
import gzip
import json
import os
import pandas as pd
import glob
from utils import Location
import pytextable as pytex

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--scores", dest="scores", help="Directory of score files")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--thres", dest="thres", type=float, help="Threshold for significance")

    args = parser.parse_args()
    
    score_files = glob.glob(os.path.join(args.scores, "*-chiasm.json.gz"), recursive=True)
    table_data = []
    # fields for the table are: testament, manuscript, condition, language, n, percentage
    for score_file in score_files:
        print(score_file)
        dir, testament, manuscript, condition, language, fname = score_file.split("/")
        with gzip.open(score_file, "rb") as ifd:
            data = [json.loads(line) for line in ifd]
    
        scores = pd.DataFrame.from_records(data)
        # print(scores.head())
        # maybe group by n and calculate the percentage of significant verses (where p < thres)
        metrics = scores.groupby("n").apply(lambda x: x["p"] < args.thres).reset_index()
        # want to know for each n value, what percentage of verses are significant
        metrics = metrics.groupby("n").mean().reset_index()
        metrics['p'] *= 100
        # add row to table
        for i, row in metrics.iterrows():
            table_data.append([testament, manuscript, condition, language, row['n'], row['p']])
    #

    # write the pytext table
    pytex.write_table(table_data, args.output)
    # df.to_csv(args.output, index=False)
    
    print("Done!")
