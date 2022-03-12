"""
will do this when I need to.
Is it absolutely necessary to keep track of idioms separately?
"""
import os
import wandb
from idiomify.fetchers import fetch_config, fetch_idioms
from idiomify.paths import ROOT_DIR


def main():
    config = fetch_config()['entities']
    # prepare idioms_df
    df = fetch_idioms(config['idioms_ver'])
    # prepare entities
    labels = [
        (f"B/{row['Idiom']}", f"I/{row['Idiom']}")
        for idx, row in df.iterrows()
    ]
    labels = [
        label
        for pair in labels
        for label in pair
    ]
    # append the "O" tag
    labels.append("O")
    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        txt_path = ROOT_DIR / "all.txt"
        with open(txt_path, 'w') as fh:
            for label in labels:
                fh.write(label + "\n")
        artifact = wandb.Artifact(name="entities", type="dataset", description=config['description'],
                                  metadata=config)
        artifact.add_file(txt_path)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        os.remove(txt_path)


if __name__ == '__main__':
    main()
