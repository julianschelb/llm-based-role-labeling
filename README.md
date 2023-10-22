# Narrative Tracking: Detecting Heroes, Villains, and Victims in Newspapers

Extracting triplets to monitor and analyze narrative shifts in news reports.

## Background

For role labeling and embedding extraction, we adopt a three-stage processing pipeline.

![Processing Pipeline](/processing_pipeline.jpg)

Three-Stage Processing Pipeline for Role Labeling and Embedding Extraction: (1) Extract hero, villain, and victims from news articles as a question answering task with Vicuna. (2) Normalize entities for spelling variations and disambiguate using GENRE with the article as context. (3) Extract embeddings with RoBERTa for clustering and further analysis.

## Installation

### Python Setup

First, create a python environment. We will use [pyenv](https://github.com/pyenv/pyenv), but other options will likely work to.

Use th following commands to (1) install a specific python version, (2) create a new virtual environment, (3) activate that environment and (4) install python dependencies.

```bash
pyenv install -v 3.10.8
pyenv virtualenv 3.10.8 hvv
pyenv activate hvv
pip install -r requirements.txt
```
## Configure Database Connection

Duplicate `.env-template` to `.env` and populate it with the appropriate database credentials.

```txt
CONNECTION_STRING="mongodb://localhost:27017/"
DATABASE_NAME="articlesDB"
```

## Run

TBD