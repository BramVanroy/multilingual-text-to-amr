import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import openai
import pandas as pd
from openai.error import RateLimitError
from openai.error import Timeout as OAITimeout
from pandas import DataFrame
from requests.exceptions import Timeout as ReqTimeout
from tqdm import tqdm
import penman

from multi_amr.data.postprocessing_graph import ParsedStatus, BACKOFF, fix_and_make_graph, \
    connect_graph_if_not_connected
from multi_amr.data.postprocessing_str import tokenize_except_quotes_and_angles

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

penman_logger = logging.getLogger("penman")
penman_logger.setLevel(logging.WARNING)

EXAMPLE_AMR_FOR_PROMPT = """This is an example AMR graph in penman notation for the English sentence "Came to study and learn". 

(c / come-01
  :purpose (a / and
             :op1 (s / study-01)
             :op2 (l / learn-01)))"""


def get_response(messages: List[Dict[str, str]], mgr_flags: DictProxy, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Post a request to the OpenAI ChatCompletion API.
    :param mgr_flags: shared manager dict to keep track of errors that we may encounter
    :param messages: a list of dictionaries with keys "role" and "content"
    :param model: the OpenAI model to use for translation
    :return: the model's translations
    """
    num_retries = 3
    last_error = None
    while num_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
        except Exception as exc:
            last_error = exc
            # If ratelimiterror on last retry, stop
            if num_retries == 1 and isinstance(exc, RateLimitError):
                mgr_flags["rate_limit_reached"] = True
                logging.exception(f"Rate limit reached on {time.ctime()}! Error trace:")
                break
            else:
                num_retries -= 1
                if isinstance(exc, (OAITimeout, ReqTimeout, TimeoutError, RateLimitError)):
                    sleep(60)
                elif isinstance(exc, openai.error.APIError):
                    sleep(30)
                else:
                    sleep(10)
        else:
            assistant_response = response["choices"][0]["message"]["content"]
            return assistant_response

    mgr_flags["total_failures"] += 1

    if last_error:
        logging.error(
            f"Error occurred on {time.ctime()}! (this is failure #{mgr_flags['total_failures']})..."
            f" Error:\n{last_error}"
        )

    return None


@lru_cache
def openai_amr_generation(
    sentence: str, fname: str, sentid: str, lang_prompt: str, mgr_flags: DictProxy, model: str = "gpt-3.5-turbo", one_shot: bool = False
) -> Dict:
    """Parse a given sentence to AMR with the OpenAI API.

    :param sentence: the input sentence
    :param lang_prompt: an explanation of the language to be used inthe prompt. Can be as simple as 'Dutch'
    :param mgr_flags: shared manager dict to keep track of errors that we may encounter
    :param model: the OpenAI model to use for translation
    :param one_shot: whether to include on example AMR tree
    :return: a (possible empty) AMR generation, or None in case of error
    """
    response = {"sentence": sentence, "fname": fname, "sentid": sentid, "uid": f"{fname}__{sentid}", "penman_str": None,
                "status": None}

    if mgr_flags["rate_limit_reached"] or mgr_flags["total_failures"] >= 3:
        return response

    if lang_prompt.lower().strip() == "english":
        system_prompt = {
            "role": "system",
            "content": f"You are an automated system that generates abstract meaning representation (AMR) from text. You must not give any explanation, comments, or notes. Your output must be structured in valid, parseable penman notation."
        }
    else:
        system_prompt = {
            "role": "system",
            "content": f"You are an automated system that generates abstract meaning representation (AMR) from text. You must not give any explanation, comments, or notes. Your output must be structured in valid, parseable penman notatio. The input text is in {lang_prompt} but your output must still follow AMR conventions, which means that the concepts and frames are more like English. This may mean that you need to translate the input text to English first, and then parse the translation to AMR.",
        }

    if one_shot:
        system_prompt["content"] += f"\n\n{EXAMPLE_AMR_FOR_PROMPT}"

    user_prompt = {
        "role": "user",
        "content": sentence,
    }

    penman_str = get_response([system_prompt, user_prompt], mgr_flags, model)

    if not penman_str or not penman_str.strip():
        return response

    penman_str = penman_str.strip()

    try:
        graph = penman.decode(penman_str)
        # Re-encode so that penman perhaps standardizes it a bit (?)
        response["penman_str"] = penman.encode(graph)
        response["status"] = ParsedStatus.OK.name.lower()
    except Exception:
        try:
            nodes = tokenize_except_quotes_and_angles(penman_str)
            graph = fix_and_make_graph(nodes, verbose=False)
        except Exception:
            response["penman_str"] = penman.encode(BACKOFF)
            response["status"] = ParsedStatus.BACKOFF.name.lower()
        else:
            try:
                graph, status = connect_graph_if_not_connected(graph)
                response["penman_str"] = penman.encode(graph)
                response["status"] = status.name.lower()
            except Exception:
                response["penman_str"] = penman.encode(BACKOFF)
                response["status"] = ParsedStatus.BACKOFF.name.lower()

    return response


def get_already_finished(df: DataFrame) -> List[str]:
    if "uid" not in df.columns:
        return []
    else:
        return df["uid"].tolist()


def translate_with_openai(
    din: str,
    fout: str,
    lang_prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_parallel_requests: int = 8,
    first_n: Optional[int] = None,
    one_shot: bool = False,
):
    openai.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")

    pdin = Path(din).resolve()
    pfout = Path(fout).resolve()

    skip_items = []
    if pfout.exists():
        if pfout.stat().st_size == 0:
            # Delete if empty
            pfout.unlink()
        else:
            df_out = pd.read_csv(pfout, sep="\t", encoding="utf-8")
            skip_items += get_already_finished(df_out)
            del df_out
    else:
        pfout.parent.mkdir(exist_ok=True, parents=True)

    data = []
    with Manager() as manager:
        mgr_flags = manager.dict({"rate_limit_reached": False, "total_failures": 0})

        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {}
            processed_items = 0
            # Submit jobs
            for pfin in pdin.rglob("*.txt"):
                with pfin.open(encoding="utf-8") as fhin:
                    for graph in penman.iterdecode(fhin):
                        sentid = graph.metadata["id"]
                        sentence = graph.metadata["snt"]
                        fname = pfin.name
                        uid = f"{fname}__{sentid}"

                        if uid in skip_items:
                            continue

                        futures[
                            executor.submit(openai_amr_generation, sentence=sentence, fname=fname, sentid=sentid, lang_prompt=lang_prompt, mgr_flags=mgr_flags, model=model, one_shot=one_shot)
                        ] = sentence
                        processed_items += 1
                        if processed_items == first_n:
                            break

                if processed_items == first_n:
                    break

            # Read job results
            failed_counter = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                result = future.result()

                penman_str = result["penman_str"]

                if penman_str is None:
                    failed_counter += 1
                    continue

                data.append(result)

            if failed_counter:
                logging.warning(f"Done processing. Had at least {failed_counter:,} failures. See the logs above.")

        if mgr_flags["rate_limit_reached"]:
            logging.error(
                "Had to abort early due to the OpenAI rate limit. Seems like you hit your limit or the server was"
                " overloaded! The generated translations have been saved. You can run the script again the continue"
                " where you left off."
            )

        if mgr_flags["total_failures"] >= 3:
            logging.error(
                "Had more than 3 catastrophic failures. Will stop processing. See the error messages above."
                " The generated translations have been saved. You can run the script again the continue"
                " where you left off."
            )

    df = pd.DataFrame(data)
    df.to_csv(pfout, sep="\t", index=False, encoding="utf-8")


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        "Generate AMR from text with OpenAI's API.\nIf you get a RateLimitError concerning using more tokens per minute"
        " than is accepted, you can try lowering --max_parallel_requests to a smaller number.\n"
        " To use this script, you need access to the OpenAI API. Make sure your API key is set as an environment"
        " variable OPENAI_API_KEY (or use --api_key). Note: THIS WILL INCUR COSTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("din", help="Input directory with text files (AMR test corpus)")
    cparser.add_argument("fout", help="Output file to write results to as tsv")
    cparser.add_argument(
        "lang_prompt",
        help="Language description to be used in the prompt. This can just be 'Dutch' but also more elaborate, e.g."
        " 'Flemish, the variant of Dutch spoken in Flanders'. If this is 'English' the system prompt will leave out"
             " instructions of pre-translating",
    )
    cparser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="Chat model to use")
    cparser.add_argument(
        "-j",
        "--max_parallel_requests",
        default=6,
        type=int,
        help="Max. parallel requests to query. Lower this if you are getting RateLimit issues.",
    )
    cparser.add_argument(
        "--api_key",
        help="OpenAI API key. By default the environment variable 'OPENAI_API_KEY' will be used, but you can override"
        " that with this option.",
    )
    cparser.add_argument(
        "--one_shot",
        action="store_true",
        help="Whether to include one example AMR structure in the system prompt.",
    )
    cparser.add_argument("-i", "--first_n", default=None, type=int, help="For debugging: only process first n items")
    translate_with_openai(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
