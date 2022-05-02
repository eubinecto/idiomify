"""
we deploy the pipeline via streamlit.
"""
import streamlit as st
from idiomifier import Idiomifier, EXAMPLE


def main():
    # fetch a pre-trained model
    idiomifer = Idiomifier()
    st.title("Idiomify with GPT-3 demo (v3.1)")
    p = st.text_area("Type a paragraph here", value=EXAMPLE)
    temp = float(st.slider(label="Creativity",
                           min_value=0.0, max_value=1.0,
                           value=0.6))
    max_tokens = st.select_slider("Maximum tokens", options=[100, 200, 300], value=300)
    if st.button(label="Idiomify"):
        if len(p.split(" ")) >= max_tokens:
            st.error(f"You can't use more than {max_tokens} tokens.")
        with st.spinner("Please wait..."):
            # highlight the rule & honorifics that were applied
            pred = idiomifer(p, temp, max_tokens)
            st.markdown(pred)


if __name__ == '__main__':
    main()
