"""
will do this when I need to.
Is it absolutely necessary to keep track of idioms separately?
"""
import os
import wandb
from idiomify.fetchers import fetch_literal2idiomatic, fetch_config
from idiomify.paths import ROOT_DIR


def main():
    config = fetch_config()['idioms']
    train_df, _ = fetch_literal2idiomatic(config['ver'])
    idioms_df = train_df[['Idiom', "Sense"]]
    idioms_df = idioms_df.groupby('Idiom').agg({'Sense': lambda x: list(set(x))})

    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        tsv_path = ROOT_DIR / "all.tsv"
        idioms_df.to_csv(tsv_path, sep="\t")
        artifact = wandb.Artifact(name="idioms", type="dataset", description=config['description'],
                                  metadata=config)
        artifact.add_file(tsv_path)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        os.remove(tsv_path)


if __name__ == '__main__':
    main()
