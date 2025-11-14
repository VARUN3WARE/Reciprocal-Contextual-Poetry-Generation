# Reciprocal Contextual Poetry Generation

This repository contains a minimal, structured implementation of PPLM + a simplified RLHF proxy adapted from the notebooks in this workspace. The goal is to provide a reproducible experiment harness that:

- Implements a simplified PPLM steering using BoW attribute models.
- Implements a lightweight RLHF proxy: a reward model + small supervised fine-tune on preferred examples.
- Runs experiments comparing Base, PPLM, RLHF, and Hybrid approaches and computes simple quantitative metrics.

Structure

- `src/` : core modules (pplm, rlhf, generator, eval)
- `scripts/` : experiment runner
- `poetry_dataset.txt` : dataset (should be present in repo root)
- `outputs/experiments/results.json` : saved experiment outputs

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the experiment runner (may download models the first time):

```bash
python scripts/run_experiments.py
```

Notes

- The RLHF implementation here is a small-scale proxy for demonstration (supervised fine-tuning on high-rated examples). Full RLHF (PPO) would need a larger training loop and compute.
- PPLM is implemented in a simplified manner for clarity and reproducibility.
- The scripts are intentionally conservative so they can run on CPU for quick tests. For larger experiments use a GPU.

Next steps

- Add unit tests and CI integration.
- Add more robust RLHF (PPO via trl) and batched PPLM for speed.
- Add plotting notebooks to visualize reward evolution and comparisons.

## Reciprocal Poetry: An AI Co-Creative Writing Partner ✍

️
This project is an interactive AI system that collaborates with a user to write poetry. It combines a fine-tuned generative model for creative suggestions with a personalized reward model that learns the user's unique stylistic preferences through real-time feedback.

(Note: You should create a short screen recording of your application in action and save it as a GIF to place here. It makes a huge difference!)

## Core Features

Fine-Tuned Generative Model: Utilizes a GPT-2 model fine-tuned on a large, cleaned corpus of classic poetry to generate high-quality, stylistically appropriate lines.

Context-Aware Generation: The AI doesn't just generate random lines. It uses the poem's initial theme and the last few lines of the ongoing poem as a rich contextual prompt. This ensures that its suggestions are coherent, relevant, and maintain a consistent flow with the user's writing.

Personalized RLHF Loop: Implements a Reinforcement Learning from Human Feedback (RLHF) system where a DistilBERT-based reward model is trained to predict the user's specific taste.

Nuanced Rating System: Instead of a simple like/dislike, users provide feedback on a 0-5 scale for multiple AI suggestions, providing rich, granular data for the reward model.

Persistent User Profiles: Learns and saves each user's preferences, allowing the AI to become more attuned to their specific style over multiple sessions.

## System Architecture

The project operates on a dual-model system: a Generator (the Poet) and a Reward Model (the Critic).

The user provides a prompt (theme and first line).

The Fine-Tuned GPT-2 (Generator) uses this context to create a set of 5 candidate lines.

The user rates all 5 candidates on a 0-5 scale.

The highest-rated line is selected and added to the poem, becoming part of the context for the next turn.

This feedback (line, rating) is stored. At the end of the session, it's used to update the DistilBERT (Reward Model), which gets better at predicting the user's preferences for the next session.

## Link for the RHLF Model

https://colab.research.google.com/drive/1aAb4D4u8u5mkvTkQG_aV_6AMwSvMyoVf#scrollTo=lDVJeRDLeoH_

## Link for the Fine-Tuned Model

https://drive.google.com/file/d/1IvTbqQ4dJ4dUDVrFT0A0Y2i1BCnsdGu0/view?usp=drive_link

## DATASET (NEW)

https://www.kaggle.com/datasets/michaelarman/poemsdataset

## Technology Stack

Backend: Python

AI/ML: PyTorch, Hugging Face Transformers

Data Handling: Pandas

Core Models: GPT-2, DistilBERT

## Future Improvements

PPLM for Form Control: Add a Plug and Play Language Model (PPLM) layer. This would allow users to enforce strict poetic forms like Haikus (5-7-5 syllables) or Sonnets (specific rhyme scheme and meter), giving them explicit, real-time structural control over the AI's generation.

Web Interface: Wrap the application in a simple web UI using Gradio or Streamlit for a more user-friendly experience.

Attention Visualization: Implement bertviz to show the user which words the model focused on in the context when generating a new line.

Larger Base Model: Fine-tune a more powerful base model (like GPT-Neo or a newer small LLM) for even higher-quality generation.

### requirements.txt file

Create a file named requirements.txt and paste the following content into it:

transformers
torch
datasets
pandas
huggingface_hub
kaggle
wordcloud
matplotlib
seaborn
