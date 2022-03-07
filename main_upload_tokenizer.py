import wandb
import shutil
from transformers import BartTokenizer
from idiomify.fetchers import fetch_config
from idiomify.paths import ROOT_DIR


def main():
    config = fetch_config()['tokenizer']
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<idiom>", "</idiom>"],  # beginning and end of an idiom
    })

    with wandb.init(entity="eubinecto", project="idiomify") as run:
        # the paths to write datasets in
        tok_dir = ROOT_DIR / "tokenizer"
        tokenizer.save_pretrained(tok_dir)
        artifact = wandb.Artifact(name="tokenizer", type="other", description=config['description'],
                                  metadata=config)
        artifact.add_dir(tok_dir)
        # then, we just log them here.
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        # don't forget to remove them
        shutil.rmtree(tok_dir)


if __name__ == '__main__':
    main()
