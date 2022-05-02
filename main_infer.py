"""
This is for just a simple sanity check on the inference.
"""
import argparse
import textwrap
import openai
from idiomifier import Idiomifier
import streamlit

openai.api_key = streamlit.secrets['OPENAI_API_KEY']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, default="I love you more than anything else.")
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=str, default=339)
    args = parser.parse_args()
    idiomifier = Idiomifier()
    res = idiomifier(args.p, args.temp, args.max_tokens)
    print("\n".join(textwrap.wrap(args.p, width=90)))
    print("\n".join(textwrap.wrap(res, width=90)))


if __name__ == '__main__':
    main()


"""
To make a successful piece of art can only be accomplished by drawing or painting what I
am truly inspired and excited to do, otherwise there is no fulfilment from completing it.
My best work has always come through being focused on the job at hand, always wanting to
improve my skill and following my instinct. My skill for creating gives me the ability to
make anything visual and bring my greatest ideas to life, no matter how obscure or
challenging they may be. When I began to really focus on my drawing, I realised the
potential I had and wanted to turn this natural talent into something more, something I
could find a career in. So I developed my drawing technique through drawing people in
different positions, under different lighting and in different environments, making sure
all proportions were right, nothing was too small or too big.
 <idiom> The proof is in the pudding </idiom>  - To make a successful piece of art can
only be accomplished by drawing or painting what I am truly inspired and excited to do,
otherwise there is no fulfilment from completing it. My best work has always come through
being focused on the job at hand, always wanting to improve my skill and following my
instinct. My skill for creating gives me the ability to make anything visual and bring my
greatest ideas to life, no matter how obscure or challenging they may be. When I began to
really focus on my drawing, I realised the potential I had and wanted to <idiom> turn my
hand </idiom> to something I could find a career in. So I developed my drawing technique
through drawing people in different positions, under different lighting and in different
environments, making sure all proportions were right, nothing was too small or too big.
"""