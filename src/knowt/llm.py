"""
Create RAG to answer questions using a VectorDB(globs=['data/corpus/**/*.txt']) of text files

>>> from knowt.constants import DATA_DIR
>>> rag = RAG(min_relevance=.5, temperature=0.05)  # (`db=DATA_DIR / 'corpus_hpr/'`)
>>> q = 'Explain phone phreaking to me'
>>> kwds = 'phreak teleph experi'.split()
>>> ans = rag.ask(q)
>>> for t in kwds:
>>>     assert t in ans.lower(), f"{t} not found in {ans.lower()}"

ans[:69] = 'Phreaking is the practice of studying, experimenting with, or explori'

>>> rag = RAG(db=DATA_DIR / 'corpus_nutrition')
>>> q = 'What is the healthiest fruit?'
>>> kwds = 'glycemic diabetes fat coconut raw'.split() + ['organic berries']
>>> ans = rag.ask(q)
>>> for t in kwds:
>>>     assert t in ans.lower(), f"{t} not found in {ans.lower()}"

ans[:69] = 'The healthiest fruits recommended for people with diabetes and glycem'

>>> q = 'How much exercise is healthiest?'
>>> kwds = 'exercis healthy'.split() + ['not specified']
>>> ans = rag.ask(q)
>>> for t in kwds:
>>>     assert t in ans.lower(), f"{t} not found in {ans.lower()}"

ans[:69] = 'The amount of exercise that is healthy for an individual is not speci'
"""
import logging
from pathlib import Path

from openai import OpenAI

from knowt.constants import OPENROUTER_API_KEY, DEFAULT_LLM_MODEL, LLAMA_MODELS_DIR
from knowt.utils import update_with_key_abbreviations

try:
    log = logging.getLogger(__name__)
except NameError:
    log = logging.getLogger("llm.__main__")


try:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
except Exception:
    CLIENT = None


# globals().update(env)
LLM_MODELS = (
    "meta-llama/llama-2-13b-chat",  # openrouter expensive
    "openai/gpt-3.5-turbo",  # openrouter free?
    "auto",  # unknown?
    "stanford-crfm/alias-gpt2-small-x21",  # openrouter <500MB
    # 'open-orca/mistral-7b-openorca',  # openrouter cheaper/better than Llama-2-13
    "Open-Orca/Mistral-7B-OpenOrca",  # huggingface
    # 'mistralai/mistral-7b-instruct',  # openrouter free
    "mistralai/Mistral-7B-Instruct-v0.2",  # huggingface
)

GGUF_MODELS = (
    "7B/llama-model.gguf",
    "models/7B/llama-model.gguf"
)

GGUF_MODEL_DICT = update_with_key_abbreviations(dict(zip(GGUF_MODELS, GGUF_MODELS)))
GGUF_MODEL_DICT.update(dict(
    llama5='llama-2-7b-chat.Q5_K_M.gguf',
    llama3='llama-2-7b-chat.Q3_K_L.gguf',
    llama2='llama-2-7b-chat.Q2_K.gguf',
))
GGUF_MODEL_DICT['llama'] = GGUF_MODEL_DICT['llama5']
GGUF_MODEL_DICT['q5'] = GGUF_MODEL_DICT['llama5']
GGUF_MODEL_DICT['q3'] = GGUF_MODEL_DICT['llama3']
GGUF_MODEL_DICT['q2'] = GGUF_MODEL_DICT['llama2']
DEFAULT_GGUF_MODEL = GGUF_MODEL_DICT['llama']


LLM_MODEL_DICT = update_with_key_abbreviations(dict(zip(LLM_MODELS, LLM_MODELS)))


PROMPT_EXAMPLES = []
PROMPT_EXAMPLES.append(
    [
        "PUG meetup",
        "2024-01-27",
        "You are an elementary school student answering questions on a reading comprehension test. "
        "Your answers must only contain information from the passage of TEXT provided. "
        "Read the following TEXT and answer the QUESTION below the text as succintly as possible. "
        "Do not add any information or embelish your answer. "
        "You will be penalized if you include information not contained in the TEXT passage. \n\n"
        "TEXT: {context}\n\n"
        "QUESTION: {question}\n\n",
    ]
)
PROMPT_EXAMPLES.append(
    [
        "Vish benchmark",
        "2024-02-01",
        "You are an elementary school student answering questions on a reading comprehension test. "
        "Your answers must only contain information from the passage of TEXT provided. "
        "Read the following TEXT and answer the QUESTION below the text as succinctly as possible. "
        "Do not add any information or embelish your answer. "
        "You will be penalized if your ANSWER includes any information not contained in the passage "
        "of TEXT provided above the QUESTION. \n\n"
        "TEXT: {context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER: ",
    ]
)
PROMPT_EXAMPLES.append(
    [
        "reading comprehension exam",
        "2024-02-12",
        "You are an elementary school student answering questions on a reading comprehension exam. \n"
        "To answer the exam QUESTION, first read the TEXT provided to see if it contains enough information to answer the QUESTION. \n"
        "Read the TEXT provided below and answer the QUESTION as succinctly as possible. \n"
        "Your ANSWER should only contain the facts within the TEXT. \n"
        "If the TEXT provided does not contain enough information to answer the QUESTION you should ANSWER with \n "
        "'I do not have enough information to answer your question.'. \n"
        "You will be penalized if your ANSWER includes any information not contained in the TEXT provided. \n\n"
        "TEXT: {context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER: ",
    ]
)
PROMPT_EXAMPLES.append(
    [
        "search results simple",
        "2024-03-26",
        "You are a personal assistant. Use the SEARCH_RESULTS provided to answer the user's question. \n"
        "Your ANSWER should only contain facts from found in the SEARCH_RESULTS. \n"
        "If the SEARCH_RESULTS text does not contain enough information to answer the QUESTION you should ANSWER with \n "
        "'The question cannot be answered based on the search results provided.'. \n"
        "SEARCH_RESULTS: {context}\n\n"
        "QUESTION: {question}\n"
        "ANSWER: ",
    ]
)
PROMPT_EXAMPLES.append(
    [
        "search results comprehension exam -- orca, hpr haycon",
        "2024-02-24",

        "You are an elementary school student answering questions on a reading comprehension exam. \n"
        "To answer the exam QUESTION, first read the SEARCH_RESULTS to see if it contains enough information to answer the QUESTION. \n"
        "Read the SEARCH_RESULTS provided below and answer the QUESTION as succinctly as possible. \n"
        "Your ANSWER should only contain facts from found in the SEARCH_RESULTS. \n"
        "If SEARCH_RESULTS text does not contain enough information to answer the QUESTION you should ANSWER with \n "
        "'The question cannot be answered based on the search results provided.'. \n"
        "You will be penalized if your ANSWER includes any information not contained in SEARCH_RESULTS. \n\n"
        "SEARCH_RESULTS: {context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER: ",
    ]
)


def get_model_path(model=DEFAULT_LLM_MODEL):
    if model not in LLM_MODEL_DICT:
        model = model.lower().strip()
    assert model in LLM_MODEL_DICT, f"{model} not found in {LLM_MODEL_DICT.keys()}"
    return LLM_MODEL_DICT.get(model, model)


def get_gguf(model=DEFAULT_GGUF_MODEL, **kwargs):
    """ Use a LLama.cpp GGUF model to generate text from your `prompt`."""
    from llama_cpp import Llama

    path = Path(model)
    if not path.is_file():
        path = LLAMA_MODELS_DIR / path
    if not path.is_file():
        path = LLAMA_MODELS_DIR / Path(model).name
    if not path.is_file():
        path = LLAMA_MODELS_DIR / Path(model).with_suffix('.gguf').name
    if not path.is_file():
        # TODO: automatically do wget  with Python:
        # wget \
        #     https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q2_K.gguf?download=true \
        #     -O ~/code/tangibleai/community/knowt/.knowt-data/llama-2-7b-Q2_K.gguf
        raise ValueError(f'Unable to find model {path.name} in {str(path.parent)}.')
    return Llama(model_path=path, **kwargs)
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window


def run_gguf(
        prompt='What questions can I ask a LLama large language model?',
        model=DEFAULT_GGUF_MODEL,
        max_tokens=80,
        stop="Q:",
        echo=False,
):
    """ Run text generation inference using LLama.cpp (GGUF) model """
    stop = [stop.rstrip(), "\n"] if isinstance(stop, str) else list(stop)
    if run_gguf.model is None:
        run_gguf.model = get_gguf(model)
    output = run_gguf.model(
        f"Q: {prompt} A: ",
        max_tokens=max_tokens,
        stop=stop,  # Stop generating just before the model would generate a new question
        echo=echo,
    )
    return output


run_gguf.model = None
