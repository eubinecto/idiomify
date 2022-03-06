"""
will do this when I need to.
Is it absolutely necessary to keep track of idioms separately?
"""
import os
import wandb
from idiomify.fetchers import fetch_literal2idiomatic, fetch_config
from idiomify.paths import ROOT_DIR


def main():
    config = fetch_config()['upload']['idioms']
    train_df, _ = fetch_literal2idiomatic(config['ver'])
    idioms = train_df['Idiom'].tolist()
    idioms = list(set(idioms))

    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        txt_path = ROOT_DIR / "all.txt"
        with open(txt_path, 'w') as fh:
            for idiom in idioms:
                fh.write(idiom + "\n")
        artifact = wandb.Artifact(name="idioms", type="dataset", description=config['description'],
                                  metadata=config)
        artifact.add_file(txt_path)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        os.remove(txt_path)


if __name__ == '__main__':
    main()
