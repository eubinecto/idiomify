from openai import Completion
import streamlit


class Idiomifier:
    """
    A few-shot ido
    """

    def __init__(self):
        self.api = Completion()

    def __call__(self, p: str, temp: float, max_tokens: int) -> str:
        # check if the request is out of limits
        prompt = f"{p}->"
        response = self.api.create(model=streamlit.secrets['ENGINE'],
                                   prompt=prompt,
                                   temperature=temp,
                                   max_tokens=max_tokens,
                                   stop="\n")
        return response.to_dict_recursive()['choices'][0]['text']
