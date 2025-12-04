#!/usr/bin/env python3
"""Interactive stepwise poem generator.

Generates short segments repeatedly and prompts the user to continue or quit.
Can load a local finetuned model or a LoRA adapter and an optional reward model.
"""
import argparse
from pathlib import Path
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generator_fixed import HybridPoetryGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='In the quiet morning,', help='Initial prompt')
    parser.add_argument('--theme', type=str, default='nature', help='Theme for PPLM steering')
    parser.add_argument('--step-length', type=int, default=20, help='Number of tokens to generate each step')
    parser.add_argument('--pplm-strength', type=float, default=0.02)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-dir', type=str, default=None, help='Optional finetuned model dir')
    parser.add_argument('--lora-dir', type=str, default=None, help='Optional PEFT/LoRA adapter dir')
    parser.add_argument('--reward-model', type=str, default=None, help='Optional reward_model.pt path')
    parser.add_argument('--out', type=str, default='outputs-gpu/experiments/generated_step.txt', help='File to append generated segments')
    parser.add_argument('--no-rlhf', action='store_true', help='Disable RLHF model usage (use base model only)')
    parser.add_argument('--no-prompt-save', action='store_true', help='Do not save generated output to disk')
    args = parser.parse_args()

    device = torch.device(args.device)
    print('Device:', device)

    base_name = 'gpt2'
    # load tokenizer/model
    if args.model_dir:
        md = Path(args.model_dir)
        if md.exists():
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(str(md))
                model = GPT2LMHeadModel.from_pretrained(str(md))
                print('Loaded model from', md)
            except Exception:
                print('Failed to load --model-dir; falling back to gpt2')
                tokenizer = GPT2Tokenizer.from_pretrained(base_name)
                model = GPT2LMHeadModel.from_pretrained(base_name)
        else:
            print('--model-dir not found; falling back to gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained(base_name)
            model = GPT2LMHeadModel.from_pretrained(base_name)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(base_name)
        model = GPT2LMHeadModel.from_pretrained(base_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # optionally load LoRA
    if args.lora_dir:
        lora_p = Path(args.lora_dir)
    else:
        lora_p = None
    if lora_p and lora_p.exists():
        try:
            from peft import PeftModel
            import json as _json
            cfg = lora_p / 'adapter_config.json'
            adapter_base = None
            if cfg.exists():
                try:
                    with open(cfg, 'r', encoding='utf-8') as f:
                        adapter_base = _json.load(f).get('base_model_name_or_path')
                except Exception:
                    adapter_base = None
            load_base = adapter_base if adapter_base is not None else base_name
            base_for_lora = GPT2LMHeadModel.from_pretrained(load_base)
            lora_model = PeftModel.from_pretrained(base_for_lora, str(lora_p))
            lora_model.to(device)
            model = lora_model
            print('Applied LoRA adapter from', lora_p)
        except Exception as e:
            print('Warning: could not load LoRA adapter:', e)

    print('Loading embedding model...')
    embed = SentenceTransformer('all-mpnet-base-v2')

    gen = HybridPoetryGenerator(model, tokenizer, embed, device)

    # load reward model if provided
    if args.reward_model:
        rm = Path(args.reward_model)
        if rm.exists():
            try:
                st = torch.load(str(rm), map_location=device)
                if hasattr(gen.reward_model, 'load_state_dict'):
                    gen.reward_model.load_state_dict(st)
                    gen.reward_model.to(device)
                    print('Loaded reward model from', rm)
            except Exception as e:
                print('Warning: failed to load reward model:', e)
        else:
            print('Provided --reward-model path does not exist; continuing without a trained reward model')
    else:
        # No reward model provided — use a tiny dummy so generator calls succeed but report N/A semantics
        class _DummyReward:
            def get_reward(self, texts, embedding_model):
                # return neutral score 0.0 for each text
                return [0.0 for _ in texts]

        gen.reward_model = _DummyReward()
        print('No --reward-model provided: using dummy reward (0.0). To get meaningful rewards, provide a trained reward model path via --reward-model')

    # prepare output file
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_prompt_save:
        outp.write_text('')

    prompt = args.prompt
    use_rlhf = not args.no_rlhf
    use_pplm = False

    print('\n=== Interactive stepwise generation ===')
    print('Controls: ENTER to continue, q + Enter to quit, p + Enter to toggle PPLM, s + Enter to save current full text, e + Enter to edit prompt')
    print('Starting prompt:')
    print(prompt)

    while True:
        # generate a short continuation (step-length tokens)
        out = gen.generate(prompt, theme=args.theme, use_rlhf=use_rlhf, use_pplm=use_pplm, pplm_strength=args.pplm_strength, max_length=args.step_length)
        full_text = out['text']
        # try to extract only the new continuation by removing the prompt prefix once
        new_piece = full_text
        try:
            if full_text.startswith(prompt):
                new_piece = full_text[len(prompt):].lstrip()
        except Exception:
            new_piece = full_text

        print('\n--- New segment ---')
        print(new_piece)
        print('--- Metrics: reward={:.6f} ---'.format(out['reward']))

        # append to file
        if not args.no_prompt_save:
            with open(outp, 'a', encoding='utf-8') as f:
                f.write(new_piece + '\n')

        # ask user for action or feedback
        try:
            cmd = input('\n[ENTER] Continue | q Quit | p Toggle PPLM | s Save full | e Edit prompt | (type feedback e.g. "i like it" or "rate 0.8") > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting interactive step generation')
            break

        cmd_l = cmd.lower()
        # control commands
        if cmd_l == 'q':
            print('Quitting.')
            break
        if cmd_l == 'p':
            use_pplm = not use_pplm
            print('PPLM now', 'ON' if use_pplm else 'OFF')
            prompt = full_text
            continue
        if cmd_l == 's':
            # save full text (prefix + saved pieces)
            with open(outp, 'a', encoding='utf-8') as f:
                f.write('\n=== SAVED FULL ===\n')
                f.write(full_text + '\n\n')
            print('Saved full text to', outp)
            prompt = full_text
            continue
        if cmd_l == 'e':
            newp = input('New prompt (empty to keep current)> ').strip()
            if newp:
                prompt = newp
                print('Prompt updated.')
            else:
                print('Keeping existing prompt.')
            continue

        # If the user entered nothing, continue (generate next step)
        if cmd == '':
            prompt = full_text
            continue

        # Otherwise treat input as feedback text or rating
        try:
            import re
            rating = None
            # numeric rating patterns: "rate 0.8" or "r 0.8" or just a number
            m = re.search(r'(?:rate|r)\s*[:=]?\s*([0-9]*\.?[0-9]+)', cmd_l)
            if m:
                try:
                    rating = float(m.group(1))
                    rating = max(0.0, min(1.0, rating))
                except Exception:
                    rating = None
            else:
                # direct numeric
                m2 = re.match(r'^([0-9]*\.?[0-9]+)$', cmd_l)
                if m2:
                    try:
                        rating = float(m2.group(1))
                        rating = max(0.0, min(1.0, rating))
                    except Exception:
                        rating = None

            # textual sentiment mapping
            if rating is None:
                pos_kw = ('like', 'love', 'good', 'great', 'nice', 'yes', 'awesome', 'ok', 'touch')
                neg_kw = ('dislike', 'hate', 'bad', 'no', 'meh', 'not')
                if any(k in cmd_l for k in pos_kw):
                    rating = 1.0
                elif any(k in cmd_l for k in neg_kw):
                    rating = 0.0

            # If we inferred a rating, register feedback and optionally update
            if rating is not None:
                print(f'Received feedback rating {rating:.3f} — adding to feedback and updating model')
                # add feedback on the full text (so the model learns from the whole generated piece)
                try:
                    gen.add_feedback(full_text, float(rating))
                except Exception as e:
                    print('Warning: failed to add feedback:', e)

                # immediate lightweight update if RLHF enabled
                if use_rlhf:
                    try:
                        gen.update_from_feedback(finetune_iters=1)
                        print('Applied lightweight update from feedback (finetune_iters=1)')
                    except Exception as e:
                        print('Warning: failed to update model from feedback:', e)

                # persist feedback to a file for traceability
                try:
                    fb_path = outp.parent / 'feedback_log.txt'
                    with open(fb_path, 'a', encoding='utf-8') as ff:
                        ff.write(f"{fb_path.name}: rating={rating:.3f} prompt={prompt}\n")
                except Exception:
                    pass

                # continue generation from updated model/context
                prompt = full_text
                continue
            else:
                # If we couldn't parse feedback, treat the text as an instruction to edit prompt
                print('Unrecognized command; treating input as new prompt')
                prompt = cmd
                continue
        except Exception as e:
            print('Error processing input; continuing. (', e, ')')
            prompt = full_text
            continue


if __name__ == '__main__':
    main()
