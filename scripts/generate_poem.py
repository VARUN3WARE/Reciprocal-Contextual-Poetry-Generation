#!/usr/bin/env python3
"""Generate a poem using the Hybrid RLHF+PPLM generator.

This script initializes models and prints/saves a hybrid-generated poem.
"""
import argparse
from pathlib import Path
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

from src.generator_fixed import HybridPoetryGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='In the quiet morning,', help='Prompt for the poem')
    parser.add_argument('--theme', type=str, default='nature', help='Theme for PPLM steering (nature|love|ocean)')
    parser.add_argument('--max-length', type=int, default=80)
    parser.add_argument('--pplm-strength', type=float, default=0.02)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, default='outputs/experiments/generated_hybrid.txt')
    parser.add_argument('--interactive', action='store_true', help='Run interactive CLI')
    args = parser.parse_args()

    device = torch.device(args.device)
    print('Using device:', device)

    print('Loading base model and tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    print('Loading embedding model...')
    embed = SentenceTransformer('all-mpnet-base-v2')

    gen = HybridPoetryGenerator(model, tokenizer, embed, device)

    def generate_and_save(prompt, theme, max_length, pplm_strength, out_path):
        print('Generating hybrid poem (RLHF + PPLM)')
        out = gen.generate(prompt, theme=theme, use_rlhf=True, use_pplm=True, pplm_strength=pplm_strength, max_length=max_length)
        poem = out['text']
        print('\n=== GENERATED POEM ===\n')
        print(poem)
        print('\n=== METRICS ===\n')
        print('Reward:', out['reward'])

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(poem)
        print('\nSaved generated poem to', out_path)

    if args.interactive:
        print('\nInteractive mode: type a prompt and press Enter (empty to quit)')
        while True:
            try:
                prompt_in = input('\nPrompt> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\nExiting interactive mode')
                break
            if not prompt_in:
                print('No prompt entered — exiting')
                break
            theme_in = input(f"Theme [{args.theme}]> ").strip() or args.theme
            pplm_in = input(f"Use PPLM? (y/N) [N]> ").strip().lower() in ('y', 'yes')
            pplm_strength_in = input(f"PPLM strength [{args.pplm_strength}]> ").strip()
            pplm_strength_in = float(pplm_strength_in) if pplm_strength_in else args.pplm_strength
            max_len_in = input(f"Max length [{args.max_length}]> ").strip()
            max_len_in = int(max_len_in) if max_len_in else args.max_length
            out_file_in = input(f"Output file [{args.out}]> ").strip() or args.out

            if pplm_in:
                generate_and_save(prompt_in, theme_in, max_len_in, pplm_strength_in, out_file_in)
            else:
                # generate without PPLM
                out = gen.generate(prompt_in, theme=theme_in, use_rlhf=True, use_pplm=False, pplm_strength=0.0, max_length=max_len_in)
                poem = out['text']
                print('\n=== GENERATED POEM ===\n')
                print(poem)
                print('\n=== METRICS ===\n')
                print('Reward:', out['reward'])
                Path(out_file_in).parent.mkdir(parents=True, exist_ok=True)
                with open(out_file_in, 'w', encoding='utf-8') as f:
                    f.write(poem)
                print('\nSaved generated poem to', out_file_in)
        return

    # non-interactive default behavior
    generate_and_save(args.prompt, args.theme, args.max_length, args.pplm_strength, args.out)


if __name__ == '__main__':
    main()
