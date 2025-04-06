import json
import re
import os
from typing import Iterable, TypedDict, TypeVar
import xml.etree.ElementTree as ET

import click
import httpx
import langfun as lf


T = TypeVar("T")
S = TypeVar("S")

_DEFAULT_MODEL_NAME = "google_genai://gemini-2.0-flash"
_DEFAULT_SEARCH_LIMIT = 100
_DEFAULT_THRESHOLD = 0.75


# Path to the folder containing this script.
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

_RELEVANCE_SCORE_PROMPT_PATH = f"{_CURR_DIR}/relevance_score_prompt.txt"
_QUALITY_SCORE_PROMPT_PATH = f"{_CURR_DIR}/quality_score_prompt.txt"


class Submission(TypedDict):
  id: str
  authors: list[str]
  title: str
  categories: list[str]
  abstract: str
  update_date: str


class RelevanceResponse(TypedDict):
  relevance_score: float
  relevance_rationale: str


class QualityResponse(TypedDict):
  quality_score: float
  quality_rationale: str


def _normalize_text(text: str) -> str:
  return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


@click.group()
def cli() -> None:
  pass


def _iter_stdin_json() -> Iterable[dict]:
  for line in click.get_text_stream("stdin"):
    if not line.strip():
      continue
    try:
      yield json.loads(line)
    except json.JSONDecodeError as exc:
      raise RuntimeError(f"Error decoding JSON: {exc}. Line: {line}") from exc


def _parse_json_response(text: str) -> dict:
  text = text.strip()
  text = text[text.find("{") : text.rfind("}") + 1]
  try:
    return json.loads(text)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Error decoding JSON: {exc}. Text: {text}") from exc


def _iter_arxiv_api(
    query: str | None = None,
    limit: int | None = None,
) -> Iterable[Submission]:
  """
  Searches arXiv using the provided query and yields results as they are processed.
  Uses httpx for making the HTTP request.

  Args:
    query: The search query string.
    limit: (Optional) The maximum number of results to retrieve.

  Yields:
    Iterable of Submission objects, where each object represents a paper.
  """

  base_url = "http://export.arxiv.org/api/query"
  limit = _DEFAULT_SEARCH_LIMIT if limit is None else limit
  params = dict(search_query=query, start=0, max_results=limit)
  response = httpx.get(base_url, params=params, timeout=300).raise_for_status()

  root = ET.fromstring(response.content)
  ns = "{http://www.w3.org/2005/Atom}"

  for entry in root.findall(f"{ns}entry"):
    author_list = entry.findall(f"{ns}author")
    category_list = entry.findall(f"{ns}category")
    yield Submission(
        id=entry.find(f"{ns}id").text.split("/")[-1],
        authors=[x.find(f"{ns}name").text for x in author_list],
        title=_normalize_text(entry.find(f"{ns}title").text),
        categories=[x.attrib["term"] for x in category_list],
        abstract=_normalize_text(entry.find(f"{ns}summary").text),
        update_date=entry.find(f"{ns}updated").text[:10],
    )


def _eval_templated_prompt(
    prompt_template: str,
    output_ctor: type[T],
    model: lf.LanguageModel,
    attachments: dict[str, str] | None = None,
    **kwargs,
) -> T:
  dl_fn = lambda x: lf.PDF.from_bytes(lf.Mime.download(x))
  attachments = {k: dl_fn(v) for k, v in (attachments or {}).items()}
  # attachments = {k: lf.Image.from_bytes(v) for k, v in attachments.items() if v is not None}
  output = lf.query(
      prompt=prompt_template,
      schema=str,
      lm=model,
      **attachments,
      **kwargs,
  )
  return output_ctor(**_parse_json_response(output))


@click.command()
@click.option("--query", default=None)
@click.option("--limit", type=int, default=None)
def search(
    query: str | None = None,
    limit: int | None = None,
) -> None:
  for submission in _iter_arxiv_api(
      query=query,
      limit=limit,
  ):
    click.echo(json.dumps(submission, ensure_ascii=False))


@click.command()
@click.option("--query", required=True)
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
def relevance(
    query: str,
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
    # qps: float | None = None,
) -> None:
  model_name = model_name or _DEFAULT_MODEL_NAME
  threshold = threshold or _DEFAULT_THRESHOLD
  lm = lf.LanguageModel.get(model_name, api_key=model_key)

  with open(_RELEVANCE_SCORE_PROMPT_PATH) as f:
    relevance_score_prompt = f.read()

  map_func = lambda record: _eval_templated_prompt(
      prompt_template=relevance_score_prompt,
      output_ctor=RelevanceResponse,
      model=lm,
      query=query,
      submission=record,
  )
  for submission, result, error in lf.concurrent_map(
      map_func,
      _iter_stdin_json(),
      max_workers=8,
      show_progress=True,
  ):
    submission: Submission
    result: RelevanceResponse
    error: Exception | None
    if error:
      raise error
    if result.get("relevance_score", 0) >= threshold:
      combined_output = dict(
          **submission,
          **result,
      )
      click.echo(json.dumps(combined_output, ensure_ascii=False))


@click.command()
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
def quality(
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
    # qps: float | None = None,
) -> None:
  model_name = model_name or _DEFAULT_MODEL_NAME
  threshold = threshold or _DEFAULT_THRESHOLD
  lm = lf.LanguageModel.get(model_name, api_key=model_key)

  with open(_QUALITY_SCORE_PROMPT_PATH) as f:
    qualitative_score_prompt = f.read()

  map_func = lambda record: _eval_templated_prompt(
      prompt_template=qualitative_score_prompt,
      output_ctor=QualityResponse,
      model=lm,
      submission=record,
      # attachments=dict(paper=f"https://ar5iv.org/pdf/{record['id']}"),
      attachments=dict(paper=f"https://arxiv.org/pdf/{record['id']}.pdf"),
  )
  for submission, result, error in lf.concurrent_map(
      map_func,
      _iter_stdin_json(),
      max_workers=8,
      show_progress=True,
  ):
    submission: Submission
    result: QualityResponse
    error: Exception | None
    if error:
      click.echo(f"Error: {error}", err=True)
      continue
    if result.get("quality_score", 0) >= threshold:
      combined_output = dict(
          **submission,
          **result,
      )
      click.echo(json.dumps(combined_output, ensure_ascii=False))


if __name__ == "__main__":
  cli.add_command(search)
  cli.add_command(relevance)
  cli.add_command(quality)
  cli()
