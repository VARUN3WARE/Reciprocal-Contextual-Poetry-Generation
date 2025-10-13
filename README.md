Reciprocal Poetry: An AI Co-Creative Writing Partner ✍️
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

## Installation
To get this project running locally, follow these steps.

1. Clone the repository:

Bash

git clone https://github.com/your-username/reciprocal-poetry.git
cd reciprocal-poetry
2. Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install the dependencies:

Bash

pip install -r requirements.txt
4. Download the Fine-Tuned Model:
Download the fine-tuned GPT-2 model (gpt2-poetry-creative.zip) from the release section or a hosting service (like Google Drive) and place it in the root of the project directory.

5. Set up Hugging Face Authentication:
The reward model downloads distilbert-base-uncased. You'll need to log in to Hugging Face.

Bash

huggingface-cli login
# Or run the notebook which has a login cell
## Usage
The project is broken into three main stages. For the final application, you only need to run step 3.

1. (Optional) Prepare the Dataset:
If you want to create the training dataset from scratch, you'll first need to download the forms.zip file from the Kaggle Dataset and place it in the root directory. Then run:

Bash

python prepare_dataset.py
This will generate a clean poetry_dataset.txt file.

2. (Optional) Fine-Tune the Model:
To fine-tune the base GPT-2 model on the dataset yourself (this requires a GPU and will take a significant amount of time):

Bash

python finetune.py
This will create a new fine-tuned model directory (e.g., gpt2-poetry-creative).

3. Run the Interactive Application:
This is the main part of the project. To start writing poetry with the AI:

Bash

python main.py
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

## License
Distributed under the MIT License. See LICENSE for more information.

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
