"""
we deploy the pipeline via streamlit.
"""
import streamlit as st
from idiomifier import Idiomifier


def check_password() -> bool:
    """
    excerpted from: https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
    Returns `True` if the user had the correct password.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["PASSWORD"] == st.secrets["PASSWORD"]:
            st.session_state["PASSWORD_CORRECT"] = True
            del st.session_state["PASSWORD"]  # don't store password
        else:
            st.session_state["PASSWORD_CORRECT"] = False

    if "PASSWORD_CORRECT" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="PASSWORD"
        )
        return False
    elif not st.session_state["PASSWORD_CORRECT"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="PASSWORD"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def main():
    if check_password():
        # fetch a pre-trained model
        idiomifer = Idiomifier()
        st.title("Idiomify with GPT-3 demo (v3.1)")
        p = st.text_area("Type a paragraph here", value="I love you more than anything else.")
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
