# Q-SOTA

Automated literature review from arXiv, using large language models as a judge to determine
relevancy and quality of papers.

## Usage

To use this, you have to clone the repo, (recommended) set up a virtual environment, and run the
python script:

```bash
git clone https://github.com/owahltinez/qsota.git && cd qsota
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
qsota <command>
```

## Models

Commands that make use of LLMs evaluate prompts via the [langfun package][langfun]. Models require
either passing an API key as an argument, or the API key can be in one of the known environment
variables i.e., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, etc.

All commands that use an LLM accept `--mode-name` and `--model-key` parameters. For a full list of
supported models, check the [langfun][langfun] documentation. Some of the models tested include:

* **Google Gemini 2.0 Flash** (`google_genai://gemini-2.0-flash`): This is the default model if no
  `--model-name` is provided.
* **GPT-4o** (`gpt-4o`): It does not support PDF format, so the quality assessment will fail unless
  the paper is available in cleartext format.
* **Claude 3.5** (`claude-3-5-sonnet-latest`): Rate limits make using this model impractical unless
  it's a very narrow search query.

[langfun]: https://pypi.org/project/langfun/

## Commands

For the exhaustive list of parameters that each command supports, run:

```bash
qsota <command> help
```

### `search`

Searches a paper database. The currently supported databases are:

* [arXiv][arxiv] (`arxiv`): [arXiv API][arxiv-api]. No API token or account required.
* [Astrophysics Data System][ads] (`ads`): [ADS API][ads-api]. Token required for authentication.

Example:

```bash
qsota search --query='wfh' --database='arxiv' > search_results.jsonl
```

The output are JSON lines, so you might want to pipe the output to a file `> search_results.jsonl`
or to one of the other commands which accept JSON lines from `stdin` as input.

You can use the optional flag `--has-doi` to filter results which have an external
[DOI](https://doi.org) associated with them. This means that the result is likely to have been
published elsewhere.

[arxiv]: https://arxiv.org/
[arxiv-api]: https://info.arxiv.org/help/api/index.html
[ads]: https://ui.adsabs.harvard.edu/
[ads-api]: https://github.com/adsabs/adsabs-dev-api

### `relevant`

Evaluates a relevancy score [0-1] for the given list of papers, and provides a rationale for the
score. This command uses **only** the title, abstract and authors of the paper to determine whether
it is relevant or not. It also requires a `query` parameter that describes, in plain words, what
is the research that the relevancy score is being evaluated for. Example:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)
cat search_results.jsonl \
    | qsota relevance --query "$CONTEXT" --threshold=0.1 \
    > relevance.jsonl
```

Once again, the output are JSON lines. So you might want to pipe the output to a file
`> relevancy.jsonl` or to one of the other commands which accept JSON lines from `stdin` as input.

### `quality`

Provides a quality score [0-1] for the given list of papers, and provides a rationale for the score.
This command will also download the full text for the paper, so it is expected to be significantly
slower. Example:

```bash
cat relevance.jsonl \
    | qsota quality --threshold=0.1 \
    > quality.jsonl
```

Once again, the output are JSON lines. So you might want to pipe the output to a file
`> quality.jsonl` or to one of the other commands which accept JSON lines from `stdin` as input.

## Chaining

The provided commands are designed to be chained together, so you can for example run:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)

qsota search --query='wfh' --database='arxiv' \
    | qsota relevance --query="$CONTEXT" --threshold=0.1 \
    | qsota quality --threshold=0.1 \
    > shortlist.jsonl
```

## Using `pipx` or `uv`

You can also install this package using [`pipx`](https://github.com/pypa/pipx):

```bash
pipx install 'git+https://github.com/owahltinez/qsota.git'
```

Alternatively, you can also use [`uv tool`](https://docs.astral.sh/uv/concepts/tools/#tools):

```bash
uv tool install 'git+https://github.com/owahltinez/qsota.git'
```

Then you can use this script as a command line tool under `qsota`:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)

qsota search --query='wfh' --database='arxiv' \
    | qsota relevance --query="$CONTEXT" --threshold=0.1 \
    | qsota quality --threshold=0.1 \
    > shortlist.jsonl
```

