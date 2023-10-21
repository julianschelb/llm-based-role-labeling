# Narrative Tracking: Detecting Heroes, Villains, and Victims in Newspapers

Extracting triplets to monitor and analyze narrative shifts in news reports.


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

Run the following scripts in order: 

* `05_fetch_articles.py`  - Download articles from the database
* `10_prepare_dataset.py` - Construct dataset by creating prompts and chunking articles
* `50_run_hvv_detection.py` - Apply generative models to detect Heroes, Villains, and Victims
* `80_upload_results.py` - Update documents in the database

**Running Scripts with nohup:** The nohup command can be used to run a command in the background and it will keep running even after you've logged out. Use nohup with your Python scripts, you would do the following:

```bash
nohup python 05_fetch_articles.py &
nohup python 10_prepare_dataset.py &
nohup python 50_run_hvv_detection.py &
nohup python 80_upload_results.py &
```