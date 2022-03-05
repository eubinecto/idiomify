"""
Here, what should you do here?
just upload all idioms here - name it as epie.
"""
import csv
import os
from idiomify.paths import ROOT_DIR
from idiomify.fetchers import fetch_pie
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="tag01")
    config = vars(parser.parse_args())

    # get the idioms here
    if config['ver'] == "tag01":
        # only the first 106, and we use this just for piloting
        literal2idiom = [
            (row[3], row[2]) for row in fetch_pie()[:106]
        ]
    else:
        raise NotImplementedError

    with wandb.init(entity="eubinecto", project="idiomify", config=config) as run:
        artifact = wandb.Artifact(name="literal2idiomatic", type="dataset")
        tsv_path = ROOT_DIR / "all.tsv"
        with open(tsv_path, 'w') as fh:
            writer = csv.writer(fh, delimiter="\t")
            for row in literal2idiom:
                writer.writerow(row)
        artifact.add_file(tsv_path)
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        os.remove(tsv_path)


if __name__ == '__main__':
    main()
