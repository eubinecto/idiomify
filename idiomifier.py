from openai import Completion
# this may take some time
ENGINE = "davinci:ft-university-of-oxford:idiomify-v3-1-2022-05-02-12-47-23"
EXAMPLE = "I love you more than anything else."


class Idiomifier:
    """
    A few-shot ido
    """

    def __init__(self):
        self.api = Completion()

    def __call__(self, p: str, temp: float, max_tokens: int) -> str:
        # check if the request is out of limits
        prompt = f"{p}->"
        response = self.api.create(engine=ENGINE,
                                   prompt=prompt,
                                   temperature=temp,
                                   max_tokens=max_tokens,
                                   end="\n")
        return response.to_dict_recursive()['choices'][0]['text']
