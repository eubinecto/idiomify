"""
literal2idiomatic ver: d-1-2
"""
import os
from idiomify.paths import ROOT_DIR
from idiomify.fetchers import fetch_pie, fetch_config
from idiomify.preprocess import upsample, cleanse, stratified_split
import wandb


def main():

    # here, we use all of them, while splitting them into train & test
    pie_df = fetch_pie()
    config = fetch_config()['literal2idiomatic']
    train_df, test_df = pie_df.pipe(cleanse)\
                              .pipe(upsample, seed=config['seed'])\
                              .pipe(stratified_split, ratio=config['train_ratio'], seed=config['seed'])
    # why don't you just "select"  the columns? yeah, stop using csv library. just select them.
    train_df = train_df[["Idiom", "Literal_Sent", "Idiomatic_Sent"]]
    test_df = test_df[["Idiom", "Literal_Sent", "Idiomatic_Sent"]]
    dfs = (train_df, test_df)
    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        train_path = ROOT_DIR / "train.tsv"
        test_path = ROOT_DIR / "test.tsv"
        paths = (train_path, test_path)
        artifact = wandb.Artifact(name="literal2idiomatic", type="dataset", description=config['description'],
                                  metadata=config)
        for tsv_path, df in zip(paths, dfs):
            df.to_csv(tsv_path, sep="\t")
            artifact.add_file(tsv_path)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        for tsv_path in paths:
            os.remove(tsv_path)


if __name__ == '__main__':
    main()
