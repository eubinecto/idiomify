"""
Right, so that was easy, then.
"""

import os
import openai
from openai.openai_object import OpenAIObject


def main():
    # Load your API key from an environment variable or secret management service
    api = openai.Completion("text-ada-001")
    response: OpenAIObject = api.create(engine="text-davinci-002",
                          prompt="Say this is a test",
                          temperature=0,
                          max_tokens=6)
    print(type(response))
    print(response.to_dict_recursive()['choices'][0]['text'])


if __name__ == '__main__':
    main()
