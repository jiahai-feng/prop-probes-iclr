from openai import OpenAI
import argparse
import coref
import os
import yaml
import openai

# imports
import random
import time
from coref.datasets.auto_gen.gpt import get_completions, scrub_quotes

def main():
    parser = argparse.ArgumentParser(description='GPT')
    parser.add_argument('--prompt', type=str, help='Path to prompt file')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='Name of model to use for completions')
    args = parser.parse_args()
    with open(args.prompt, 'r') as f:
        rewrite_prompt = f.read()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": rewrite_prompt},
    ]
    client = OpenAI()

    output = []
    for i in range(args.num_samples):
        response = get_completions(
            client=client,
            model=args.model_name,
            messages=messages
        )
        output.append(scrub_quotes(response.choices[0].message.content))
    with open(args.output_file, 'w') as f:  
        yaml.dump(output, f)

if __name__ == '__main__':
    main()