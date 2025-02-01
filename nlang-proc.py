#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import signal

options = {
    'llama-cli': '~/Programming/llama.cpp/build/bin/llama-cli',
    'model': '~/Programming/llama.cpp/models/models/Mistral-Nemo-Instruct-2407.Q6_K.gguf',
    'n_tokens': -1,
    'context': 12000,
    'ngl': 99,
    'no_cnv': True
}

def get_prompt(data, instruction):
    return f"""[INST]The following is a string of textual data that is to be processed.[/INST]

[DATA]
{data}
[/DATA]

[INST]
{instruction}
[/INST]

[OUT]"""

def generate(data, instruction):
    command = f"""{options['llama-cli']} \
            -m {options['model']} \
            -ngl {options['ngl']} -c {options['context']} -n {options['n_tokens']} \
            --no-display-prompt \
            -r '[/OUT]' \
            --prompt '{get_prompt(data, instruction)}'"""
    if options['no_cnv']:
        command += " -no-cnv"

    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = '0'

    process = subprocess.Popen(command, text=True, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def signal_handler(sig, frame):
        process.send_signal(signal.SIGKILL)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    prevChunk = ""
    while True:
        chunk = process.stdout.read(8)
        if not chunk:
            print(prevChunk, end='', flush=True)
            break
        buffer = prevChunk + chunk
        if '[/OUT]' in buffer:
            print(buffer.split('[/OUT]')[0], end='', flush=True)
            break
        print(prevChunk, end='', flush=True)
        prevChunk = chunk

    process.stdout.close()
    process.wait()

def main():
    parser = argparse.ArgumentParser(description='Process some natural language data.')
    parser.add_argument('input', nargs='*', help='Input strings to be processed')
    parser.add_argument('--llama-cli', type=str, help='Path to llama-cli executable')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--n_tokens', type=int, help='Number of tokens to generate')
    parser.add_argument('--context', type=int, help='Context size')
    parser.add_argument('--ngl', type=int, help='Number of layers to put on the GPU')

    args = parser.parse_args()

    if args.llama_cli:
        options['llama-cli'] = args.llama_cli
    if args.model:
        options['model'] = args.model
    if args.n_tokens:
        options['n_tokens'] = args.n_tokens
    if args.context:
        options['context'] = args.context
    if args.ngl:
        options['ngl'] = args.ngl

    input_string = ' '.join(args.input)

    # Read stdin data into `data`
    data = ''
    while True:
        try:
            data += input() + '\n'
        except EOFError:
            break

    generate(data, input_string)

if __name__ == "__main__":
    main()