"""
literal2idiomatic ver: d-1-2
"""
import os
from idiomify.paths import ROOT_DIR
from idiomify.fetchers import fetch_pie, fetch_config
from idiomify.preprocess import upsample, cleanse, stratified_split, replace_labels
import wandb


def main():

    # here, we use all of them, while splitting them into train & test
    pie_df = fetch_pie()
    config = fetch_config()['literal2entities']
    train_df, test_df = pie_df.pipe(cleanse)\
                              .pipe(upsample, seed=config['seed'])\
                              .pipe(replace_labels)\
                              .pipe(stratified_split, ratio=config['train_ratio'], seed=config['seed'])
    # why don't you just "select"  the columns? yeah, stop using csv library. just select them.
    train_df = train_df[["Idiom", "Sense", "Literal_Sent", "Literal_Label"]]
    test_df = test_df[["Idiom", "Sense", "Literal_Sent", "Literal_Label"]]
    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        train_path = ROOT_DIR / "train.tsv"
        test_path = ROOT_DIR / "test.tsv"
        artifact = wandb.Artifact(name="literal2entities", type="dataset", description=config['description'],
                                  metadata=config)
        train_df.to_csv(train_path, sep="\t")
        test_df.to_csv(test_path, sep="\t")
        artifact.add_file(train_path)
        artifact.add_file(test_path)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        os.remove(train_path)
        os.remove(test_path)


if __name__ == '__main__':
    main()
