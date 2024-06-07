# constants.py
import datetime
import logging
import os

from pathlib import Path

log = logging.getLogger(__name__)


def set_loglevel():
    global DEBUG, LOGLEVEL, logging, env, log
    try:
        import dotenv  # noqa - dotenv is installed with `knowt[all]` or `knowt[llm]`

        dotenv.load_dotenv()
        env, log = set_loglevel()  # running again, in case .env file overrides DEBUG or LOGLEVEL env vars
    except Exception as exc:
        log.info(str(exc))
        log.warning(
            "Unable to import `dotenv` or no `.env` file found.\n"
            "The dotenv package is not needed unless you plan to use external LLM services."
        )
    env = dict(os.environ)
    DEBUG = env.get("DEBUG", "").strip()
    LOGLEVEL = env.get("LOGLEVEL", "WARNING").strip()
    if DEBUG and DEBUG.lower()[0] in "tyd1":
        DEBUG = True
        LOGLEVEL = "DEBUG"
    else:
        DEBUG = ""
    LOGLEVELS = dict(zip("diwef", "DEBUG INFO WARNING ERROR FATAL".split()))
    if LOGLEVEL:
        LOGLEVEL = LOGLEVELS.get(LOGLEVEL.lower()[0], "")

    if LOGLEVEL:
        logging.basicConfig(level=getattr(logging, LOGLEVEL))
    try:
        log = logging.getLogger(__name__)
    except NameError:
        log = logging.getLogger("llm.__main__")

    return env, log


env, log = set_loglevel()


try:
    # Will only work
    BASE_DIR = Path(__file__).parent.parent.parent
    assert (BASE_DIR / "src").is_dir()
except Exception:
    BASE_DIR = Path.home() / ".knowt-data"

DATA_DIR = env.get('DATA_DIR', BASE_DIR / ".knowt-data")
DATA_DIR.mkdir(exist_ok=True)
CORPUS_HPR = env.get('CORPUS_HPR', DATA_DIR / "corpus_hpr")
CORPUS_NUTRITION = env.get('CORPUS_NUTRITION', DATA_DIR / "corpus_nutrition")
CORPUS_DIR = env.get('CORPUS_DIR', CORPUS_HPR)
CORPUS_DIR.mkdir(exist_ok=True)
LLAMA_MODELS_DIR = DATA_DIR / 'llama' / 'models'
LLAMA_MODELS_DIR.mkdir(exist_ok=True, parents=True)
LLAMA_MODELS_URL_TEMPLATE = 'https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/{}?download=true'


OPENROUTER_API_KEY = env.get("OPENROUTER_API_KEY", None)
DEFAULT_LLM_MODEL = env.get('DEFAULT_LLM_MODEL', 'orca')
DEFAULT_SPACY_MODEL = env.get('DEFAULT_SPACY_MODEL', 'en_core_web_sm')
DEFAULT_ENCODER = env.get('DEFAULT_ENCODER', 'spacy')
GLOBS = env.get('GLOBS', ("**/*.txt", "**/*.md"))  # add preprocessors for ReST and HTML

TEXT_LABEL = env.get('TEXT_LABEL', "sentence")
DF_PATH = env.get('DF_PATH', CORPUS_DIR / f"{TEXT_LABEL}s.csv.bz2")
EMBEDDINGS_PATH = env.get('EMBEDDINGS_PATH', DF_PATH.with_suffix(".embeddings.joblib"))
EXAMPLES_PATH = env.get('EXAMPLES_PATH', DF_PATH.with_suffix(".search_results.csv"))

TODAY = env.get('TODAY', datetime.date.today())

DEFAULT_LLM_MODEL = env.get('DEFAULT_LLM_MODEL', 'Orca')
