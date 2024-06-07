""" Language models for tokenizing, sentencizing and embedding (encoding) natural language text:
* SpaCy (en_core_web_sm)
* BERT (all-MiniLM-L6-v2)
"""

import spacy
import time
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from knowt.constants import TEXT_LABEL, DEFAULT_ENCODER, DEFAULT_SPACY_MODEL
from knowt.utils import update_with_key_abbreviations

log = logging.getLogger(__name__)

MODEL_NAMES = dict(
    spacy='en_core_web_sm',
    bert='all-MiniLM-L6-v2',
)
SPACY_MODELS = [f"en_core_web_{siz}" for siz in 'sm md lg'.split()]
SPACY_MODELS = update_with_key_abbreviations(dict(zip(SPACY_MODELS, SPACY_MODELS)), seps=['_'], positions=[0, -1])
MODEL_NAMES.update(SPACY_MODELS)
MODEL_NAMES = update_with_key_abbreviations(MODEL_NAMES)

# raise ValueError(
#     'FIXME: Need to validate embedding model dimensionality and be able to load multiple db objects from multiple joblib files.')


def load_spacy_model(model=None, disable=["ner", "parser"], tokenizer_only=None, **kwargs):
    model = model or 'en_core_web_sm'
    if tokenizer_only and (disable is None or not len(disable) or not disable):
        disable = ["ner", "parser"]
    try:
        nlp = spacy.load(model, disable=disable)
    except OSError:
        spacy.cli.download(model)
        nlp = spacy.load(model, disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    return nlp


spacy_model_name = DEFAULT_SPACY_MODEL
nlp = load_spacy_model()  # need a tokenizer, even if spacy embedding vectors are not used


def construct_encoder(model, *args, show_progress_bar=True, **kwargs):
    """ Factory for building encoder models with .encode methods: [str] -> [[]] (Hugging Face transformers API) """
    global nlp
    if not model:
        return construct_encoder(DEFAULT_ENCODER, *args, **kwargs)
    if isinstance(model, str):
        model = MODEL_NAMES.get(model, model)  # normalize the name of the encoder model
    if not isinstance(model, (str, bytes)) and hasattr(model, 'encode') and callable(model.encode):
        model_encode = model.encode
        return model_encode
    if model in SPACY_MODELS:
        nlp = load_spacy_model(model, *args, **kwargs)

        def spacy_encode(iterable_of_strings):
            if show_progress_bar:
                iterable_of_strings = tqdm(iterable_of_strings)
            vectors = []
            for s in iterable_of_strings:
                vectors.append(nlp(s).vector)
            return np.array(vectors)
        return spacy_encode
    try:
        return construct_encoder(SentenceTransformer(model, *args, **kwargs))

    except Exception as exc:
        log.warning(exc)


def encode_dataframe(df, encoder=None, columns=[TEXT_LABEL]):
    """Generate embedding vectors, one for each row in the DataFrame.

    Embeddings are concatenated columnwise:
    >>> e1 = np.arange(6).reshape(2,3)
    >>> e1
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> e2 = np.arange(2, 14, 2).reshape(2,3)
    >>> e2
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> np.array(np.concatenate([e2, e2], axis=1))
    array([[ 0,  1,  2,  2,  4,  6],
           [ 3,  4,  5,  8, 10, 12]])
    """
    encoder = construct_encoder(encoder)
    if isinstance(columns, (str, int)):
        columns = [columns]
    log.info(
        f"Generating embeddings for {len(df)}x{len(columns)} documents (sentences)..."
    )
    embeddings = []
    for i, col in enumerate(columns):
        log.debug('Generating embeddings for column #{i}/{len(df)}, "{col}".')
        if isinstance(col, int):
            col = df.columns[col]
        num_tokens = df["num_tokens"].sum()
        num_chars = df[col].str.len().sum()
        t = time.time()
        embeddings.append(encoder(df[col].tolist()))
        t = time.time() - t
        log.info(
            f"Finished embedding in column {i}:{col} with {len(df)} documents ({num_tokens} tokens) in {t} s."
        )
        log.info(
            f"   {num_tokens/t/1000:.6f} token/ms\n   {num_chars/t/1000:.6f} char/ms"
        )

    return np.array(embeddings).T


def generate_passages(
    fin, min_line_len=0, max_line_len=1024, batch_size=1
):
    """Crude rolling window chunking of text file: width=max_line_len, stride=max_line_len//2"""
    for line in fin:
        # Break up long lines into overlapping passages of text
        if len(line) > max_line_len:
            stride = max_line_len // 2
            for text in generate_passages(
                [line[i: i + max_line_len] for i in range(0, stride, len(line))]
            ):
                yield text
        yield line


def generate_batches(iterable, batch_size=1000):
    """Break an iterable into batches (lists of len batch_size containing iterable's objects)"""
    generate_batches.batch = []
    # TODO: preallocate numpy array with batch_size and increment i, truncating last numpy array
    for obj in iterable:
        generate_batches.batch.append(obj)
        if len(generate_batches.batch) >= batch_size:
            yield generate_batches.batch
            generate_batches.batch = []


generate_batches.batch = []


def generate_all_embeddings(df, encoder=DEFAULT_ENCODER, text_label=TEXT_LABEL, show_progress_bar=True):
    """Generate embedding vectors, one for each row in the DataFrame."""
    if isinstance(encoder, str) and not getattr(generate_all_embeddings, 'encoder_name', '') == encoder:
        setattr(generate_all_embeddings, 'encoder_name', encoder)
        encoder = construct_encoder(encoder, show_progress_bar=show_progress_bar)
        setattr(generate_all_embeddings, 'encoder', encoder)
    log.info(f"Generating embeddings for {len(df)} documents (sentences)...")
    num_tokens = df["num_tokens"].sum()
    num_chars = df[text_label].str.len().sum()
    t = time.time()
    embeddings = encoder(df[text_label].tolist())
    t -= time.time()
    t *= -1
    log.info(f"Finished embedding {len(df)} sentences({num_tokens} tokens) in {t} s.")
    log.info(f"   {num_tokens/t/1000:.6f} token/ms\n   {num_chars/t/1000:.6f} char/ms")
    return embeddings
