from transformers import BartTokenizer, BartForConditionalGeneration

from data import IdiomifyDataModule


CONFIG = {
    "literal2idiomatic_ver": "pie_v0",
    "batch_size": 20,
    "num_workers": 4,
    "shuffle": True
}


def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    datamodule = IdiomifyDataModule(CONFIG, tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        srcs, tgts_r, tgts = batch
        input_ids, attention_mask = srcs[:, 0], srcs[:, 1]  # noqa
        decoder_input_ids, decoder_attention_mask = tgts_r[:, 0], tgts_r[:, 1]
        outputs = bart(input_ids=input_ids,
                       attention_mask=attention_mask,
                       decoder_input_ids=decoder_input_ids,
                       decoder_attention_mask=decoder_attention_mask)
        logits = outputs[0]
        print(logits.shape)
        """
        torch.Size([20, 47, 50265])
        (N, L, |V|)
        """

        break


if __name__ == '__main__':
    main()
