from openai import Completion

# --- constants --- #
with open("./prompt.txt", 'r') as fh:
    PROMPT = fh.read()

EXAMPLE = "I love you more than anything else."


class Idiomifier:
    """
    A few-shot ido
    """

    def __init__(self):
        self.api = Completion()

    def __call__(self, engine: str, p: str, temp: float, max_tokens: int) -> str:
        # check if the request is out of limits
        prompt = f"{PROMPT}\n{p}\n->"
        response = self.api.create(engine=engine,
                                   prompt=prompt,
                                   temperature=temp,
                                   max_tokens=max_tokens)
        return response.to_dict_recursive()['choices'][0]['text']
