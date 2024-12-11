import json
import os
from typing import Iterable, TypedDict, TypeVar

import click
import kagglehub
import llm

T = TypeVar("T")
S = TypeVar("S")

_DEFAULT_THRESHOLD = 0.75


# Path to the folder containing this script.
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

_DATASET_PUBLISHER = "Cornell-University"
_DATASET_HANDLE = "arxiv"
_DATASET_SUBPATH = "arxiv-metadata-oai-snapshot.json"

_RELEVANCE_SCORE_PROMPT_PATH = f"{_CURR_DIR}/relevance_score_prompt.txt"
_QUALITY_SCORE_PROMPT_PATH = f"{_CURR_DIR}/quality_score_prompt.txt"


class Submission(TypedDict):
  id: str
  authors: str
  title: str
  doi: str
  categories: str
  abstract: str
  update_date: str


@click.group()
def cli() -> None:
  pass


@click.command()
def kaggle_login() -> None:
  try:
    kagglehub.whoami()
  except Exception as exc:
    click.echo(f"Error: {exc}")
    kagglehub.login()


@click.command()
@click.option("--force-download", is_flag=True)
def refresh(force_download: bool = False) -> str:
  # Check if kagglehub has the appropriate credentials set.
  local_path = kagglehub.dataset_download(
      handle=f"{_DATASET_PUBLISHER}/{_DATASET_HANDLE}",
      path=_DATASET_SUBPATH,
      force_download=force_download,
  )
  return local_path


def _iter_stdin_json() -> Iterable[dict]:
  for line in click.get_text_stream("stdin"):
    if not line.strip():
      continue
    try:
      yield json.loads(line)
    except json.JSONDecodeError as exc:
      raise RuntimeError(f"Error decoding JSON: {exc}. Line: {line}") from exc


def _iter_submissions(
    date_start: str | None = None,
    date_end: str | None = None,
    limit: int | None = None,
) -> Iterable[Submission]:
  with open(refresh.callback()) as f:
    for line in f:
      record: Submission = json.loads(line)
      record["title"] = record["title"].replace("\n", "")
      if date_start and record["update_date"] < date_start:
        continue
      if date_end and record["update_date"] > date_end:
        continue
      yield record

      if limit is not None:
        limit -= 1
        if limit <= 0:
          break


def _format_prompt(prompt: str, **kwargs) -> str:
  for key, value in kwargs.items():
    if isinstance(value, dict):
      value = json.dumps(value)
    prompt = prompt.replace("{{ " + key + " }}", str(value))
  return prompt


def _parse_json_response(text: str) -> dict:
  text = text.strip()
  text = text[text.find("{") : text.rfind("}") + 1]
  try:
    return json.loads(text)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Error decoding JSON: {exc}. Text: {text}") from exc


def _eval_templated_prompt(
    prompt_substitutions: dict,
    model: llm.Model,
    prompt_template: str,
    attachments: list[llm.Attachment] | None = None,
) -> dict:
  prompt = _format_prompt(prompt_template, **prompt_substitutions)
  response = model.prompt(prompt, attachments=attachments)
  return dict(
      **_parse_json_response(response.text()),
      **prompt_substitutions,
  )


@click.command()
@click.option("--date-start", default=None)
@click.option("--date-end", default=None)
@click.option("--limit", type=int, default=None)
def search(
    date_start: str | None = None,
    date_end: str | None = None,
    limit: int | None = None,
) -> None:
  for submission in _iter_submissions(
      date_start=date_start,
      date_end=date_end,
      limit=limit,
  ):
    click.echo(json.dumps(submission))


@click.command()
@click.option("--query", required=True)
@click.option("--model", default=None)
@click.option("--threshold", type=float, default=None)
def relevance(
    query: str,
    model: str | None = None,
    threshold: float | None = None,
) -> None:
  threshold = threshold or _DEFAULT_THRESHOLD
  llm_model = llm.get_model(model)

  with open(_RELEVANCE_SCORE_PROMPT_PATH) as f:
    relevance_score_prompt = f.read()

  for record in _iter_stdin_json():
    result = _eval_templated_prompt(
        prompt_substitutions=dict(query=query, **record),
        model=llm_model,
        prompt_template=relevance_score_prompt,
    )
    if result.get("relevance_score", 0) >= threshold:
      click.echo(json.dumps(result))


@click.command()
@click.option("--model", default=None)
@click.option("--threshold", type=float, default=None)
def quality(
    model: str | None = None,
    threshold: float | None = None,
) -> None:
  threshold = threshold or _DEFAULT_THRESHOLD
  llm_model = llm.get_model(model)

  with open(_QUALITY_SCORE_PROMPT_PATH) as f:
    qualitative_score_prompt = f.read()

  for record in _iter_stdin_json():
    pdf_url = f"https://arxiv.org/pdf/{record['id']}"
    result = _eval_templated_prompt(
        prompt_substitutions=record,
        model=llm_model,
        prompt_template=qualitative_score_prompt,
        attachments=[llm.Attachment(url=pdf_url)],
    )
    if result.get("quality_score", 0) >= threshold:
      click.echo(json.dumps(result))


if __name__ == "__main__":
  cli.add_command(kaggle_login)
  cli.add_command(refresh)
  cli.add_command(search)
  cli.add_command(relevance)
  cli.add_command(quality)
  cli()
