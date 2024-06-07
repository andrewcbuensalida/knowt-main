"""
# TODO: Specialized MemMap file for Wikipedia articles

# 1. memmap.MemMap in VectorDB
- use_memmap=False __init__ kwarg
- use memmap within VectorDB instead of RAM CSV
- run doctests
- add speedtests

# 2. Specialized Wikipedia MemMap
- fix wikipedia page download (consider using `pwiki` instead of nlpia2-wikipedia)
- add binary category tags to all pages downloaded
- script to convert MariaDB *.sql.gz to CSVs
- add category prefix to all titles https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-category.sql.gz
- add page title to intro paragraph

>>> db = VectorDB(DATA_DIR / 'corpus_nutrition', search_limit=10, min_relevance=.5)
>>> q = 'What is the healthiest fruit?'
>>> topdocs = db.search(q, min_relevance=.69)
>>> len(topdocs)
3
>>> topdocs[TEXT_LABEL].iloc[0][:30]
'Fruit -- not too much at all. '
"""

import argparse
import jsonlines as jsl
import logging
import re
import sys
import time

import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from knowt.constants import DATA_DIR, CORPUS_DIR, GLOBS, DF_PATH, TEXT_LABEL

from knowt.embedding import nlp, construct_encoder

log = logging.getLogger(__name__)


# raise ValueError(
#     'FIXME: Need to validate embeddings.py and joblib dimensionality and be able to load multiple db objects from multiple joblib files.'
# )


def load_text_files(
    corpus_dir=CORPUS_DIR,
    globs=GLOBS,
    text_label=TEXT_LABEL,
    max_files=None,
    max_docs=None,
    limit=None,
    show_progress_bar=True,
):
    """Load text files into strings, split into sentences, and return a DataFrame of those sentences."""
    globs = globs or GLOBS
    if isinstance(globs, str):
        globs = (globs,)
    corpus_dir = Path(corpus_dir)
    filepaths = []
    if corpus_dir.is_file():
        filepaths.append(corpus_dir)
        corpus_dir = corpus_dir.parent
    elif corpus_dir.is_dir():
        for g in globs:
            log.info(f'glob: {g}')
            filepaths.extend(list(corpus_dir.glob(g)))
    filepaths = sorted(filepaths)

    df = []
    enumerate_filepaths = enumerate(filepaths)
    if show_progress_bar:
        enumerate_filepaths = tqdm(enumerate_filepaths)
    for i, path in enumerate_filepaths:
        if limit and i >= limit:
            break
        log.info(f"Processing {path.name}...")
        with open(path, "r", encoding="utf-8") as fin:
            text = fin.read()

        # Find the index of all line endings and spans for all file lines
        line_starts = {0: 1}
        # FIXME this should be a one-liner re.findall(s) or s.index()
        for i, char in enumerate(text):
            if char == "\n":
                line_starts[i + 1] = line_starts[i] + 1
            else:
                line_starts[i + 1] = line_starts[i]

        doc = nlp(text)
        for sent in doc.sents:
            start_char = sent.start_char
            line_number = line_starts[start_char]
            text = sent.text.strip()
            df.append({
                "path": path,
                "filename": path.name,
                text_label: " ".join(text.splitlines()) or " ",
                "line_number": line_number,
                "line_start": line_starts[start_char],
                "sent_start_char": start_char,
                "len": len(sent.text),
                "num_tokens": len(sent),
            })

    return pd.DataFrame(df)


def prettify_search_results(top_docs, ascending=False, limit=10, text_label=TEXT_LABEL):
    """Print out a sorted list of search result sentence pairs with relevance"""
    # TODO: highlight most relevant parts of sentencess with **text**
    if not len(top_docs):
        return "No results found."
    top_docs = top_docs.sort_values("relevance", ascending=ascending)
    return "\n".join(
        [
            f"{i+1:02d} {row['relevance']:.2f}: {row[text_label]}\n"
            + f"    ({Path(row['filename']).name}:{row['line_number']})"
            for i, (index, row) in enumerate(top_docs.iterrows())
            if i < limit
        ]
    )


def pretty_print(top_docs):
    if isinstance(top_docs, pd.DataFrame):
        top_docs = prettify_search_results(top_docs, ascending=False)
    if isinstance(top_docs, (list, tuple)):
        top_docs = "\n".join(top_docs)
    print(top_docs)


class VectorDB:
    def __init__(
        self,
        df=DF_PATH,
        embeddings=None,
        encoder='bert',  # 'spacy', 'tfidf', 'trigram'
        search_limit=None,
        max_files=None,
        max_docs=None,  # a document can be a sentence, so might not be entire file
        min_relevance=0,
        refresh=False,
        text_label=TEXT_LABEL,
        show_progress_bar=True,
    ):
        self.show_progress_bar = True
        self.search_limit = search_limit or 10
        self.max_files = max_files
        self.max_docs = max_docs
        self.min_relevance = min_relevance
        self.encoder_name = str(encoder)
        log.warning(self.encoder_name)
        self.encoder = construct_encoder(encoder)
        log.warning(self.encoder)
        self.df_path = None
        self.num_dim = len(self.encoder(["hi"])[0])
        self.text_label = text_label = text_label or TEXT_LABEL

        if isinstance(df, VectorDB):
            self.__dict__.update(df.__dict__)

        # df should be either a .csv.bz2 file Path or a DataFrame:
        if isinstance(df, (str, Path)):
            df = Path(df)
            if df.is_dir():
                df = df / f"{text_label}s.csv.bz2"
            elif (DATA_DIR / df).is_dir():
                df = DATA_DIR / df
                df = DATA_DIR / df / f"{text_label}s.csv.bz2"
            if df.is_file():
                log.warning(f"Loading existing index from {df}...")
            elif df.parent.is_dir():
                log.warning(
                    f"Unable to find document index at {df} so creating new index for "
                    f"{self.max_files} files ({self.max_docs} docs) in {df.parent}.")
                refresh = True
            else:
                raise ValueError(
                    f"ERROR: unable to load {text_label}s from {str(df.parent)[:256]} of type {type(df)}."
                )
            self.df_path = df
            self.df_paths = [self.df_path]  # TODO: store all paths here? don't assume parent is corpus (full path in sentences.csv)
            if self.df_path.is_file():
                df = pd.read_csv(self.df_path, index_col=None)
        df[text_label] = df[text_label].fillna("")

        self.df_path = Path(self.df_path or DF_PATH)
        self.embeddings_path = self.df_path.parent / (self.df_path.name.split(".")[0] + ".embeddings.joblib")
        log.debug(f"embeddings_path: {self.embeddings_path}")
        if embeddings is None and self.embeddings_path.is_file():
            log.info(f"Loading embeddings from file {self.embeddings_path}...")
            self.embeddings = joblib.load(self.embeddings_path)
            if self.embeddings.shape != (len(df), self.num_dim):
                self.embeddings = None

        if refresh or not isinstance(df, pd.DataFrame):
            self.df_path = Path(self.df_path or df or DF_PATH)
            log.warning(f"Indexing text files at {self.df_path.parent}")
            df = load_text_files(self.df_path.parent, max_files=self.max_files, max_docs=self.max_docs)
            log.warning(f"Writing new index DataFrame to {self.df_path}")
            df.to_csv(self.df_path, index=False)

        if isinstance(embeddings, (pd.DataFrame, np.ndarray)):
            self.embeddings = np.array(embeddings)
            log.debug(f"1 (DataFrame) embeddings type: {type(self.embeddings)}")
            joblib.dump(self.embeddings_path)
        elif (
            refresh
            or not isinstance(self.embeddings_path, (str, Path))
            or not self.embeddings_path.is_file()
            or self.embeddings is None
        ):
            self.embeddings = self.generate_all_embeddings(df)
            log.debug(f"2 (refresh) embeddings type: {type(self.embeddings)}")
            log.debug(f"2 embeddings_path: {self.embeddings_path}")
            joblib.dump(self.embeddings, self.embeddings_path)
        self.df = df

    def __add__(self, added_db):
        added_db = VectorDB(added_db)
        self.df = pd.concat([self.df, added_db.df], axis=0, ignore_index=True)
        self.embeddings = pd.concat([self.embeddings, added_db.embeddings], axis=0, ignore_index=True)
        self.max_docs = None
        self.max_files = None
        self.df_paths = [self.df_path, added_db.df_path]

    def search(self, query, min_relevance=None, limit=None, n_sentences=2):
        """Search the corpus sentence embeddings with the equivalent embedding of the query string."""
        if isinstance(limit, int) and limit > 0:
            self.search_limit = limit
        limit = limit or self.search_limit

        if isinstance(min_relevance, (float, int)):
            self.min_relevance = min_relevance
        if min_relevance is None:
            min_relevance = self.min_relevance
        if min_relevance is True:
            min_relevance = self.min_relevance
        min_relevance = min_relevance or 0

        if not isinstance(self.embeddings, pd.DataFrame):
            self.embeddings = pd.DataFrame(self.embeddings)
        self.n_sentence_embeddings = self.embeddings.rolling(n_sentences).mean()
        query_embedding = self.encoder([query])
        similarities = cosine_similarity(
            query_embedding, self.n_sentence_embeddings[n_sentences - 1:]
        )[0]
        top_indices = np.argsort(similarities)[-limit:]
        top_docs_df = self.df.iloc[top_indices].copy()
        top_docs = [self.df.iloc[top_indices + i] for i in range(n_sentences)]
        top_docs_df[self.text_label] = [
            " ".join(s) for s in zip(*[d[self.text_label] for d in top_docs])
        ]
        top_docs_df["relevance"] = similarities[top_indices].copy()

        return top_docs_df[top_docs_df["relevance"] >= min_relevance].copy()

    def generate_all_embeddings(self, df=None, in_place=True, max_docs=None):
        """Generate embedding vectors, one for each row in the DataFrame and return it as DataFrame."""
        df = self.df if df is None else df
        if in_place:
            self.df = df
        if max_docs is None:
            max_docs = len(df)
        log.info(f"Generating embeddings for {len(df)} documents (sentences)...")
        num_tokens = df["num_tokens"].sum()
        num_chars = df[self.text_label].str.len().sum()
        t0 = time.time()
        embeddings = self.encoder(df[self.text_label].tolist())
        delta_t = time.time() - t0
        log.info(f"Finished embedding {len(df)} sentences({num_tokens} tokens) in {delta_t} s.")
        log.info(f"   {num_tokens/delta_t/1000:.6f} tokens/ms\n   {num_chars/delta_t/1000:.6f} char/ms")
        if in_place:
            self.num_tokens = num_tokens
            self.num_chars = num_chars
        return embeddings

    def preprocess_query(self, query):
        """Transform questions into statements likely to correspond to useful sentences in the corpus."""
        query = query.strip()
        is_question = ""
        if query[-1] == "?":
            is_question = query
            query = query.rstrip("?")
        query_doc = nlp(query)
        if query_doc[0].text.lower() in "who what when where why how":
            is_question = is_question or query
            suffix = f"{query_doc[0].text}."
            query = query_doc[1:].text
            if query_doc[1].pos_ in "AUX VERB":
                query = query_doc[2:].text + f"{query_doc[1].text} {suffix}"
        return query

    def search_pretty(self, query, preprocess=True, min_relevance=None, limit=None):
        """Return search results as a string rather than returning a DataFrame"""
        if isinstance(limit, int) and limit > 0:
            self.search_limit = limit
        limit = limit or self.search_limit
        if preprocess is True:
            preprocess = self.preprocess_query
        if callable(preprocess):
            query = preprocess(query)
        top_docs = self.search(query, min_relevance=min_relevance, limit=limit)
        return prettify_search_results(top_docs, limit=self.search_limit)

    def cli(self, preprocess=True):
        while True:
            query = input("Search query ([ENTER] to exit)")
            if query.lower().strip() in ("exit", "exit()", "quit", "quit()", ""):
                break
            if preprocess is True:
                preprocess = self.preprocess_query
            if callable(preprocess):
                query = preprocess(query)
            top_docs = self.search(query)
            self.pretty_print(top_docs)


def with_suffixes(path, suffixes, sep="."):
    """Strip all dotted suffixes from a filepath obj.

    >>> p = with_suffixes('/dir1/dir2/what.ever.suf1.suf2.suf3', '')
    >>> str(p)
    '/dir1/dir2/what'
    >>> p = with_suffixes('/dir1/dir2/what.ever.suf1.suf2.suf3', '.txt', sep='.')
    >>> str(p)
    '/dir1/dir2/what.txt'
    >>> p = with_suffixes('/dir1/dir2/what.ever.suf1.suf2.suf3', 'txt')
    >>> str(p)
    '/dir1/dir2/what.txt'
    >>> p = with_suffixes('/dir1/dir2/what_ever_suf1.suf2.suf3', 'date.txt', sep='_')
    >>> p.name
    'what_date.txt'
    """
    if not isinstance(suffixes, str):
        suffixes = sep + sep.join(suffixes)
    suffixes = sep + suffixes if suffixes and suffixes[0] != sep else suffixes
    path = Path(path)
    return path.parent / (path.name.split(sep)[0] + suffixes)


# def basepath(path):
#     path = Path(path)
#     return path.parent / path.name.split('.')[0]


def update_votes(df, votes, filepath=DATA_DIR / "votes.jsonl"):
    with jsl.open(filepath, mode="a") as writer:
        for v in votes:
            writer.write(
                dict(
                    vote=v,
                )
            )


def main(df_path=None, embeddings=None, refresh=False):
    argv = sys.argv[1:]
    i = 0
    flags = []
    votes = []
    for i, a in enumerate(argv):
        if a[0] not in "-+":
            break
        m = re.match(r"[-+]\d", a)
        if m:
            votes.append(float(a))
            continue
        flags.append(a)
    args = argv[i:]
    # args = parser.parse_args(flags)
    query = " ".join(args)
    if not len(query):
        query = input("AMA: ")

    df_path = Path(df_path or DF_PATH)
    log.info(
        f"Starting script with corpus {df_path}, embeddings {embeddings}, and refresh={refresh}."
    )

    embeddings_path = with_suffixes(embeddings or df_path, ".embeddings.joblib")
    embeddings_path.parent.mkdir(exist_ok=True)
    df_path.parent.mkdir(exist_ok=True)
    db = VectorDB(df=df_path, embeddings=embeddings_path, refresh=refresh)
    print(db.search_pretty(query))
    # db.cli()
    return db


parser = argparse.ArgumentParser(
    prog="search",
    description='A command line semantic search engine and RAM "database".',
    epilog="by Ethan Cavill with support from Hobson Lane",
)
# parser.add_argument("corpus_dir")
parser.add_argument("-r", "--refresh", action="store_true")
parser.add_argument("-c", "--cache", default=DF_PATH)
for i in range(1, 11):
    parser.add_argument(f"-{i}", f"--downvote{i}", action="store_true", default=None)

if __name__ == "__main__":
    db = main()
    #     db.pretty_search()
