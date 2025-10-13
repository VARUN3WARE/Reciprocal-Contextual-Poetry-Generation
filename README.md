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

## Link for the Fine-Tuned Model

https://drive.google.com/file/d/1IvTbqQ4dJ4dUDVrFT0A0Y2i1BCnsdGu0/view?usp=drive_link



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
