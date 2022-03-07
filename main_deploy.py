"""
we deploy the pipeline via streamlit.
"""
import re
import streamlit as st
from idiomify.fetchers import fetch_config, fetch_idiomifier, fetch_idioms, fetch_tokenizer
from idiomify.pipeline import Pipeline


@st.cache(allow_output_mutation=True)
def fetch_resources() -> tuple:
    config = fetch_config()['idiomifier']
    model = fetch_idiomifier(config['ver'])
    tokenizer = fetch_tokenizer(config['tokenizer_ver'])
    idioms = fetch_idioms(config['idioms_ver'])
    return config, model, tokenizer, idioms


def main():
    # fetch a pre-trained model
    config, model, tokenizer, idioms = fetch_resources()
    model.eval()
    pipeline = Pipeline(model, tokenizer)
    st.title("Idiomify Demo")
    text = st.text_area("Type sentences here",
                        value="Just remember that there will always be a hope even when things look hopeless")
    with st.sidebar:
        st.subheader("Supported idioms")
        idioms = [row["Idiom"] for _, row in idioms.iterrows()]
        st.write(" / ".join(idioms))

    if st.button(label="Idiomify"):
        with st.spinner("Please wait..."):
            sents = [sent for sent in text.split(".") if sent]
            preds = pipeline(sents, max_length=200)
            # highlight the rule & honorifics that were applied
            preds = [re.sub(r"<idiom>|</idiom>", "`", pred)
                     for pred in preds]
            st.markdown(". ".join(preds))


if __name__ == '__main__':
    main()
