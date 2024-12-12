import asyncio
import functools
import json
import re
import os
import time
from typing import Callable, Iterable, TypedDict, TypeVar
import xml.etree.ElementTree as ET

import click
import httpx
import llm
import tqdm.auto as tqdm


T = TypeVar("T")
S = TypeVar("S")

_DEFAULT_SEARCH_LIMIT = 100
_DEFAULT_THRESHOLD = 0.75

_MODEL_API_QPS = {
    "gemini-2.0-flash-exp": 5 / 60,
    "gemini-1.5-flash-latest": 2_000 / 60,
    "gemini-1.5-pro-latest": 1_000 / 60,
}
_DEFAULT_MODEL_API_QPS = 8


# Path to the folder containing this script.
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

_RELEVANCE_SCORE_PROMPT_PATH = f"{_CURR_DIR}/relevance_score_prompt.txt"
_QUALITY_SCORE_PROMPT_PATH = f"{_CURR_DIR}/quality_score_prompt.txt"


class Submission(TypedDict):
  id: str
  authors: str
  title: str
  categories: str
  abstract: str
  update_date: str


class RateLimitedAsyncModel(llm.AsyncModel):

  def rate_limited_prompt(
      self,
      prompt: str,
      *,
      attachments: list[llm.Attachment] | None = None,
  ) -> asyncio.Future[llm.AsyncResponse]:
    raise NotImplementedError


def _normalize_text(text: str) -> str:
  return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _rate_limited_func(
    func: Callable[[T], asyncio.Future[S]],
    qps: float,
    semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
) -> Callable[[T], asyncio.Future[S]]:
  @functools.wraps(func)
  async def wrapper(*args, **kwargs):
    async with semaphore:
      # Get the start time (monotonic clock) before calling the function.
      start_time = time.monotonic()
      # Call the function with the provided arguments.
      result = func(*args, **kwargs)
      # Get the end time (monotonic clock) after calling the function.
      end_time = time.monotonic()
      # Calculate the time taken to call the function.
      elapsed_time = end_time - start_time
      # Calculate the time to sleep to maintain the desired QPS.
      sleep_time = max(0, 1 / qps - elapsed_time)
      # Sleep for the calculated time.
      await asyncio.sleep(sleep_time)
      # Return the result of the function call.
      return result

  return wrapper


async def _future_and_data(
    future: asyncio.Future[T],
    data: S,
) -> asyncio.Future[tuple[T, S]]:
  return await future, data


def coroutine(func: Callable[..., T]) -> Callable[..., T]:
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    return asyncio.run(func(*args, **kwargs))
    # yield from func(*args, **kwargs)

  return wrapper


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
    yield dict(
        id=entry.find(f"{ns}id").text.split("/")[-1],
        authors=[x.find(f"{ns}name").text for x in author_list],
        title=_normalize_text(entry.find(f"{ns}title").text),
        categories=[x.attrib["term"] for x in category_list],
        abstract=_normalize_text(entry.find(f"{ns}summary").text),
        update_date=entry.find(f"{ns}updated").text[:10],
    )


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


def _init_model(
    model_name: str | None = None,
    model_key: str | None = None,
    qps: float | None = None,
) -> RateLimitedAsyncModel:
  # Load the model and set the key if provided.
  llm_model = llm.get_async_model(model_name)
  if model_key:
    llm_model.key = model_key

  # Enforce API limits for the model.
  llm_model.rate_limited_prompt = _rate_limited_func(
      llm_model.prompt,
      qps=qps or _MODEL_API_QPS.get(model_name, _DEFAULT_MODEL_API_QPS),
  )

  return llm_model


async def _download_attachment(
    attachment: llm.Attachment,
) -> llm.Attachment:
  async with httpx.AsyncClient() as client:
    res = await client.get(
        attachment.url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    attachment.type = res.headers.get("Content-Type") or "text/plain"
    attachment.content_bytes = res.content
    return attachment


async def _eval_templated_prompt(
    prompt_substitutions: dict,
    model: RateLimitedAsyncModel,
    prompt_template: str,
    attachments: list[llm.Attachment] | None = None,
) -> dict:
  # Download the attachments ourselves.
  download_tasks = [_download_attachment(a) for a in attachments or []]
  attachments = await asyncio.gather(*download_tasks)

  # Distinguish between text and non-text attachments.
  text_attachments = [x for x in attachments if x.type.startswith("text/")]
  other_attachments = [x for x in attachments if not x.type.startswith("text/")]

  # Embed the text attachments directly into the prompt, keep others as-is.
  texts = [x.content_bytes.decode() for x in text_attachments]
  prompt_substitutions = dict(
      prompt_substitutions,
      text_attachments="\n\n".join(texts),
  )

  prompt = _format_prompt(prompt_template, **prompt_substitutions)
  response = await model.rate_limited_prompt(
      prompt,
      attachments=other_attachments,
  )
  return _parse_json_response(await response.text())


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
    click.echo(json.dumps(submission))


@click.command()
@click.option("--query", required=True)
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
@click.option("--qps", type=float, default=None)
@coroutine
async def relevance(
    query: str,
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
    qps: float | None = None,
) -> None:
  threshold = threshold or _DEFAULT_THRESHOLD
  llm_model = _init_model(model_name, model_key, qps)

  with open(_RELEVANCE_SCORE_PROMPT_PATH) as f:
    relevance_score_prompt = f.read()

  tasks: list[asyncio.Task[tuple[dict, Submission]]] = []
  for record in _iter_stdin_json():
    result = _eval_templated_prompt(
        prompt_substitutions=dict(query=query, **record),
        model=llm_model,
        prompt_template=relevance_score_prompt,
    )
    tasks.append(asyncio.ensure_future(_future_and_data(result, record)))

  for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
    result, record = await task
    if result.get("relevance_score", 0) >= threshold:
      click.echo(json.dumps(dict(**result, **record), ensure_ascii=False))


@click.command()
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
@click.option("--qps", type=float, default=None)
@coroutine
async def quality(
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
    qps: float | None = None,
) -> None:
  threshold = threshold or _DEFAULT_THRESHOLD
  llm_model = _init_model(model_name, model_key, qps)

  with open(_QUALITY_SCORE_PROMPT_PATH) as f:
    qualitative_score_prompt = f.read()

  tasks: list[asyncio.Task[tuple[dict, Submission]]] = []
  for record in _iter_stdin_json():
    # url = f"https://arxiv.org/pdf/{record['id']}"
    url = f"https://ar5iv.org/html/{record['id']}"
    result = _eval_templated_prompt(
        prompt_substitutions=record,
        model=llm_model,
        prompt_template=qualitative_score_prompt,
        attachments=[llm.Attachment(url=url)],
    )
    tasks.append(asyncio.ensure_future(_future_and_data(result, record)))

  for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
    try:
      result, record = await task
      if result.get("quality_score", 0) >= threshold:
        click.echo(json.dumps(dict(**result, **record), ensure_ascii=False))
    except ValueError as exc:
      click.echo(f"Error: {exc}", err=True)


if __name__ == "__main__":
  cli.add_command(search)
  cli.add_command(relevance)
  cli.add_command(quality)
  cli()
