from transformers import BartTokenizer
from idiomify.datamodules import IdiomifyDataModule


CONFIG = {
    "literal2idiomatic_ver": "pie_v0",
    "batch_size": 20,
    "num_workers": 4,
    "shuffle": True
}


def main():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    datamodule = IdiomifyDataModule(CONFIG, tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        srcs, tgts_r, tgts = batch
        print(srcs.shape)
        print(tgts_r.shape)
        print(tgts.shape)


if __name__ == '__main__':
    main()
