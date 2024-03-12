# %% [markdown]
# # Stage 1: Role Labeling

# %% [markdown]
# One way to identify narratives in newspaper text is through considering the character archetypes relied on to compose the framing of an article. The main figures in an article may be represented as the heroes, villains, or victims in the text to guide the reader towards reading the article in context with existing qualities implicit in these character archetypes. Gomez-Zara et al present a dictionary-based method for computationally determining the hero, villain, and victim in a newspaper text, which Stammbach et al adapt by using an LLM for the same task.

# %% [markdown]
# ## Fetch Articles (for Testing)

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient
from transformers import BertTokenizer
from utils.preprocessing import *
from utils.accelerators import *
from utils.multithreading import *
from utils.database import *
from utils.model import *
from utils.files import *
from datasets import Dataset
from rouge import Rouge
from tqdm import tqdm
import statistics
import hashlib
import random
import openai
import time
import math
import re

# %% [markdown]
# ### Connect to Database

# %% [markdown]
# Credentials are sourced from the `.env` file.

# %%
_, db = getConnection(use_dotenv=True)

# %% [markdown]
# ***

# %% [markdown]
# ## Load Model

# %% [markdown]
# Vicuna-13B is an open-source chatbot developed by refining LLaMA through user-contributed conversations gathered from ShareGPT. Initial assessments employing GPT-4 as a referee indicate that Vicuna-13B attains over 90%* quality of OpenAI ChatGPT and Google Bard, surpassing other models such as LLaMA and Stanford Alpaca in over 90%* of instances.
#
# See:
# * https://github.com/lm-sys/FastChat
# * https://huggingface.co/lmsys/vicuna-13b-v1.5-16k

# %% [markdown]
# ```bash
# # Start the controller service
# nohup python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001 &
#
# # Start the model_worker service
# nohup python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5-16k --num-gpus 2 &
#
# # Start the gradio_web_server service
# nohup python3 -m fastchat.serve.gradio_web_server --host 0.0.0.0 --port 7860 &
#
# # Launch the RESTful API server
# nohup python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8080 &
# ```

# %% [markdown]
# Check GPU utilization:

# %%
num_gpus = torch.cuda.device_count()
print(f'Number of available GPUs: {num_gpus}')

# %% [markdown]
# List infos about the available GPUs:

# %%
gpu_info_list = listAvailableGPUs()

# %%

# %% [markdown]
# Test Model:

# %%
model = RemoteModel(model_name="vicuna-13b-v1.5-16k",
                    api_base="http://merkur22.inf.uni-konstanz.de:8080/v1",
                    api_key="EMPTY")

params = {'n': 5}
result = model.generateAnswers("Once upon a time", params=params)
print(result, type(result))

# %% [markdown]
# ***

# %% [markdown]
# ## Define Prompt Template:

# %%
# PROMPT_TEMPLATE = "Please identify entities which are portrayed as hero, villain and victim in the following news article. A hero is an individual, organisation, or entity admired for their courage, noble qualities, and outstanding achievements. A villain is a character, organisation, or entity known for their wickedness or malicious actions, often serving as an antagonist in a story or narrative. A victim is an individual, organisation, or entity who suffers harm or adversity, often due to an external force or action. Every entity can only be one of those roles. The solution must be returned in this format {{hero: \"Name\", villain: \"Name\", victim: \"Name\"}}. Article Headline: ''{headline}''. Article Text: ''{article_text}''  Solution: "

# PROMPT_TEMPLATE = "Please identify entities which are portrayed as hero, villain and victim in the following news article. Every entity can only be one of those roles. If not existing return None as name. The solution must be returned in this format {{hero: \"Name\", villain: \"Name\", victim: \"Name\"}}. Article Headline: ''{headline}''. Article Text: ''{article_text}''  Solution: "

# PROMPT_TEMPLATE = "Please identify entities which are portrayed as hero, villain and victim in the following news article. Each entity can only assume one role. If none apply, use 'None'. The solution must be returned in this format {{hero: \"Name\", villain: \"Name\", victim: \"Name\"}}. Article Headline: ''{headline}''. Article Text: ''{article_text}''  Solution: "

PROMPT_TEMPLATE = "Given the news article below, identify entities categorized as a hero, villain, or victim. Each entity can only assume one role. If none apply, use 'None'. The solution must be provided in this format: {{hero: \"Name\", villain: \"Name\", victim: \"Name\"}}. \n Headline: '{headline}' \n Text: '{article_text}' \n Solution: "

# Test the template with a dummy text
prompt_test = PROMPT_TEMPLATE.format(headline='Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
                                     article_text='Lorem ipsum dolor sit amet, consectetur adipiscing elit.')
print(prompt_test)


# %% [markdown]
# ## Define Parameter for Text Generation

# %% [markdown]
# Each parameter influences the text generation in a specific way. Below are the parameters along with a brief explanation:
#
# **`max_length`**:
# * Sets the maximum number of tokens in the generated text (default is 50).
# * Generation stops if the maximum length is reached before the model produces an EOS token.
# * A higher `max_length` allows for longer generated texts but may increase the time and computational resources required.
#
# **`min_length`**:
# * Sets the minimum number of tokens in the generated text (default is 10).
# * Generation continues until this minimum length is reached even if an EOS token is produced.
#
# **`num_beams`**:
# * In beam search, sets the number of "beams" or hypotheses to keep at each step (default is 4).
# * A higher number of beams increases the chances of finding a good output but also increases the computational cost.
#
# **`num_return_sequences`**:
# * Specifies the number of independently computed sequences to return (default is 3).
# * When using sampling, multiple different sequences are generated independently from each other.
#
# **`early_stopping`**:
# * Stops generation if the model produces the EOS (End Of Sentence) token, even if the predefined maximum length is not reached (default is True).
# * Useful when an EOS token signifies the logical end of a text (often represented as `</s>`).
#
# **`do_sample`**:
# * Tokens are selected probabilistically based on their likelihood scores (default is True).
# * Introduces randomness into the ƒgeneration process for diverse outputs.
# * The level of randomness is controlled by the 'temperature' parameter.
#
# **`temperature`**:
# * Adjusts the probability distribution used for sampling the next token (default is 0.7).
# * Higher values make the generation more random, while lower values make it more deterministic.
#
# **`top_k`**:
# * Limits the number of tokens considered for sampling at each step to the top K most likely tokens (default is 50).
# * Can make the generation process faster and more focused.
#
# **`top_p`**:
# * Also known as nucleus sampling, sets a cumulative probability threshold (default is 0.95).
# * Tokens are sampled only from the smallest set whose cumulative probability exceeds this threshold.
#
# **`repetition_penalty`**:
# * Discourages the model from repeating the same token by modifying the token's score (default is 1.5).
# * Values greater than 1.0 penalize repetitions, and values less than 1.0 encourage repetitions.
#

# %%
params = {'do_sample': True,
          'early_stopping': True,
          # 'max_length': 100,
          # 'min_length': 1,
          'logprobs': 1,
          'n': 3,
          # 'best_of': 1,

          'num_beam_groups': 2,
          'num_beams': 5,
          'num_return_sequences': 5,
          'max_tokens': 50,
          'min_tokens': 0,
          'output_scores': True,
          'repetition_penalty': 1.0,
          'temperature': 0.6,
          'top_k': 50,
          'top_p': 1.0
          }

# %% [markdown]
# ## Define Helper Functions

# %%


def extractTriplet(answer):
    """ Extracts the triplet from the answer string. """

    # Extract keys and values using regex
    keys = re.findall(r'(\w+):\s*\"', answer)
    values = re.findall(r'\"(.*?)\"', answer)
    result = dict(zip(keys, values))

    if result == {}:
        keys = re.findall(r'(\w+):\s*([^,]+)', answer)
        result = dict((k, v.strip('"')) for k, v in keys)

    return result

# %%


def getAnswersTriplets(article, model, template, params):
    """ Generates answers for the given article using the model and template. """

    # Extract the article headline and text
    article_headline = article.get("title", "")
    article_text = article.get("parsing_result").get("text")

    # Generate the answer
    prompt = template.format(headline=article_headline,
                             article_text=article_text)
    answers = model.generateAnswers(prompt, params)

    return answers

# %%


def splitText(text, n_tokens, tokenizer, overlap=10):
    """Splits the input text into chunks with n_tokens tokens using HuggingFace tokenizer, 
    with an overlap of overlap tokens from the previous and the next chunks."""

    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0

    # No previous chunk at the beginning, so no need for overlap
    chunks.append(tokenizer.convert_tokens_to_string(tokens[i:i+n_tokens]))
    i += n_tokens

    while i < len(tokens):
        # Now, we include overlap from the previous chunk
        start_index = i - overlap
        end_index = start_index + n_tokens
        chunk = tokens[start_index:end_index]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        i += n_tokens - overlap  # Moving the index to account for the next overlap

    return chunks

# %%


def processBatch(articles, model, template, params, chunk_size=1024, overlap=256, show_progress=False, verbose=False, fist_chunk_only=False):
    """Processes a batch of articles and extracts the triplets."""
    runtimes = []  # List to store the runtime for each article

    # Iterate over the articles
    for article in tqdm(articles, desc="Generating answers", disable=not show_progress):

        try:

            start_time = time.time()  # Start the timer

            # Extract the article headline and text
            article_headline = article.get("title", "")
            article_text = article.get("parsing_result").get("text")

            if fist_chunk_only:
                chunks = splitText(article_text, chunk_size,
                                   model.tokenizer, overlap=0)
                print("Chunks:", len(chunks))
                print(chunks)
                chunks = [chunks[0]]
            else:
                # Split the article text into chunks
                chunks = splitText(article_text, chunk_size,
                                   model.tokenizer, overlap=overlap)

                # print("Chunks:", len(chunks))

            chunk_results = []
            for chunk_id, chunk in enumerate(chunks):

                if verbose:
                    print("Chunk:", chunk_id)
                    print("Chunk Length:", calcInputLength(
                        model.tokenizer, chunk))

                prompt = template.format(
                    headline=article_headline, article_text=chunk)
                answers = model.generateAnswers(prompt, params)

                # Extract the triplet from seach answer
                for answer in answers:
                    answer["triplet"] = extractTriplet(answer.get("text"))

                results = {
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                    "answers": answers
                }
                chunk_results.append(results)

            article["triplets"] = chunk_results

            end_time = time.time()  # End the timer
            runtime = end_time - start_time  # Calculate the runtime
            runtimes.append(runtime)  # Store the runtime
        except Exception as e:
            print("Error:", e)

    return articles, runtimes

# %%


def updateArticle(db, id: str, values: dict = {}, collection="articles"):
    "Updates scraping task in database"
    filter = {"_id": ObjectId(id)}
    values = {"$set": {**values}}
    r = db[collection].update_one(filter, values)
    return r

# %%


def updateArticles(db, articles, collection="articles"):
    """Updates the articles in the database."""

    for article in tqdm(articles, desc="Uploading results"):
        id = article.get("_id")
        values = {"triplets": article.get("triplets", [])}
        updateArticle(db, id, values, collection)

# %% [markdown]
# ## Make Predictions


# %%
LIMIT = 100  # Number of articles to process in each batch
CHUNK_SIZE = 1024  # Number of tokens in each chunk
OVERLAP = 64  # Number of overlapping tokens between chunks
COLLECTION = "articles.sampled.triplets"

FIELDS = {"url": 1, "title": 1, "parsing_result.text": 1}
# QUERY = {"triplets": {"$exists": False},
#          "parsing_result.word_count": {"$gt": 1300, "$lt": 200_000_000}}
# QUERY = {"triplets": {"$exists": False}}
QUERY = {"triplets.0.answers": {"$size": 0}}

# %%
batch_id = 0

while True:
    print(f"------ Batch {batch_id} ------")

    # Fetch the next batch of articles
    articles = fetchArticleTexts(db, LIMIT, 0, FIELDS, QUERY, COLLECTION)

    # Stop if no more articles are available
    if not articles:
        print("No more articles available")
        break

    # Process the batch of articles
    articles, runtimes = processBatch(articles, model, PROMPT_TEMPLATE,
                                      params, chunk_size=CHUNK_SIZE, overlap=OVERLAP, show_progress=True, verbose=True, fist_chunk_only=True)

    # Update the articles in the database
    updateArticles(db, articles, COLLECTION)
    print(f"Updated {len(articles)} articles", end="\n\n")

    batch_id += 1
