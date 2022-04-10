"""
Right, so that was easy, then.
"""

import os
import openai


def main():
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(engine="text-davinci-002",
                                        prompt="Say this is a test",
                                        temperature=0,
                                        max_tokens=6)
    print(response)


if __name__ == '__main__':
    main()
