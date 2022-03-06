"""
we deploy the pipeline via streamlit.
"""
from typing import Tuple, List
import streamlit as st
from transformers import BartTokenizer
from idiomify.fetchers import fetch_config, fetch_idiomifier, fetch_idioms
from idiomify.pipeline import Pipeline
from idiomify.models import Idiomifier


@st.cache(allow_output_mutation=True)
def fetch_resources() -> Tuple[dict, Idiomifier, BartTokenizer, List[str]]:
    config = fetch_config()['idiomifier']
    model = fetch_idiomifier(config['ver'])
    idioms = fetch_idioms(config['idioms_ver'])
    tokenizer = BartTokenizer.from_pretrained(config['bart'])
    return config, model, tokenizer, idioms


def main():
    # fetch a pre-trained model
    config, model, tokenizer, idioms = fetch_resources()
    model.eval()
    pipeline = Pipeline(model, tokenizer)
    st.title("Idiomify Demo")
    st.markdown(f"Author: `Eu-Bin KIM`")
    st.markdown(f"Version: `{config['ver']}`")
    text = st.text_area("Type sentences here",
                        value="Just remember there will always be a hope even when things look black")
    with st.sidebar:
        st.subheader("Supported idioms")
        st.write(" / ".join(idioms))

    if st.button(label="Idiomify"):
        with st.spinner("Please wait..."):
            sents = [sent for sent in text.split(".") if sent]
            sents = pipeline(sents, max_length=200)
            # highlight the rule & honorifics that were applied
            st.write(". ".join(sents))


if __name__ == '__main__':
    main()
