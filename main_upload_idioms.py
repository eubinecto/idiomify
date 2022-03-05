"""
Here, what should you do here?
just upload all idioms here - name it as epie.
"""
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
        # only the first 106, and this is for piloting
        idioms = set([row[0] for row in fetch_pie()[:106]])
    else:
        raise NotImplementedError
    idioms = list(idioms)

    with wandb.init(entity="eubinecto", project="idiomify", config=config) as run:
        artifact = wandb.Artifact(name="idioms", type="dataset")
        txt_path = ROOT_DIR / "all.txt"
        with open(txt_path, 'w') as fh:
            for idiom in idioms:
                fh.write(idiom + "\n")
        artifact.add_file(txt_path)
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        os.remove(txt_path)


if __name__ == '__main__':
    main()
