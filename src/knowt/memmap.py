import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from knowt.constants import DATA_DIR
from knowt.embedding import model, generate_batches

NUM_DIM = len(model.encode(["Hello"])[0])


class MemMap:
    """FIXME: Wrapper for .memmap tables of vectors, pairs vectors with string labels (index)."""

    def __init__(self, path, ncols=384, start=1, stop=None, mode="r+", order="C"):
        raise NotImplementedError
        self.path = Path(path)
        self.ncols = ncols
        self.mmpath = path.parent / path.name + ".memmap"

        # self.mm = np.memmap(
        #     self.mmpath,
        #     mode=mode,
        #     # offset=0,
        #     # dtype=np.float32,
        #     order=order
        # )

        # # FIXME: move to MemMap.create() -- only required if using generators to create memmap files
        # self.start = 1
        # self.stop = stop
        # self.fin = iter(self.path.open())
        # self.header = []
        # while start:
        #     start -= 1
        #     self.header += [next(self.fin)]
        # self.nrows = sum(1 for _ in self.path.open())
        # if self.stop:
        #     self.nrows = min(stop - start, self.nrows)
        # self.it = iter(self.path.open())

        df = pd.read_csv(self.path)
        if len(df.columns) > 1:
            df = df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()])
        self.labels = df[df.columns[0]].values
        self.vectors = df[list(df.columns)[1:]].values
        assert self.labels.dtype == np.dtype(
            str
        ), f"Invalid dtype for labels (first column) ({self.labels.dtype})"
        assert self.labels.dtype in (
            np.float32,
            np.int,
            np.float16,
            np.float64,
        ), f"Invalid dtype for vectors (all columns after first one) ({self.vectors.dtype})"


# def generate_encodings(batches, encoder=model.encode, batch_size=1, **kwargs):
#     """ `for encoding_batch in generate_encodings(generate_batches(generate_passages(fin)))` encoding_batch.dot(whatever)"""
#     for batch in generate_passages(batches, **kwargs):
#         yield encoder(batch)  # [t for t in enumerate()])


def encode_to_memmap(
    fin,
    fout=None,
    skip=0,
    batch_size=500,
    preprocessor=lambda x: x.strip().replace("_", " "),
    encoder=model.encode,
):
    encoder = getattr(encoder, "encode", encoder)
    num_dim = len(encoder(["hi"])[0])
    if isinstance(fin, str):
        if "\n" not in fin and "\r" not in fin:
            fin = Path(fin)
        else:
            fin = io.StringIO(fin)  # first line of string must be a filename!
    if isinstance(fin, Path):
        fin = fin.open()

    fout = fout or getattr(fin, "name", None)
    print(fout)
    fout = fout or DATA_DIR / fin.readline().strip()
    print(fout)
    fout = Path(fout).with_suffix(".memmap")
    print(fout)
    if isinstance(fout, str):
        fout = Path(fout)
    iterable_passage_batches = iter(generate_batches(fin))  # generate_passages(fin)))
    try:
        mm = np.memmap(fout, mode="r+", offset=0, dtype=np.float32, order="C")
        mm.reshape(len(mm) // num_dim, num_dim)
        mm.flush()
        print(mm)
    except Exception as e:
        print(type(e), e)
        vectors = model.encode(next(iterable_passage_batches))
        print(vectors.shape)
        mm = np.memmap(
            fout,
            mode="w+",
            offset=0,
            shape=vectors.shape,
            dtype=vectors.dtype,
            order="C",
        )
        mm[:] = vectors.copy()
        mm.flush()
        print(mm)
    # docs = np.loadtxt(fin, dtype=str)
    for batch_num, batch in tqdm(enumerate(iterable_passage_batches)):
        print(batch)
        vectors = model.encode(batch).copy()
        print(vectors.shape)
        mm.resize(mm.shape[0] + len(vectors), mm.shape[1])
        mm[-len(vectors) :] = vectors
        mm.flush()
    mm.flush()
    return mm


def write_memmap(
    fin, fout, it=None, mm=None, skip=1, dtype=np.float32, num_dim=384, batch_size=1000
):
    """DUPLICATES encode_to_memmap()"""
    for i in range(skip):
        fin.readline()
    num_rows = sum(1 for _ in fin)
    fin.seek(0)
    for i in range(skip):
        fin.readline()
    it = it or iter(
        generate_batches(
            iter(x.strip().replace("_", " ") for x in fin), batch_size=1000
        )
    )
    mm = np.memmap(
        fout, mode="w+", offset=0, shape=(num_rows, num_dim), dtype=dtype, order="C"
    )
    for i, batch in tqdm(enumerate(it), total=num_rows // batch_size):
        mm[i * batch_size : (i + 1) * batch_size, :] = model.encode(batch)
    mm.flush()


def find_similar(memmap, limit=1, queries=None, labels=None, encoder=model.encode):
    if isinstance(memmap, (str, Path)):
        memmap = np.memmap(memmap, mode="r+")
    if isinstance(queries, str):
        queries = [queries]
    num_docs, num_dims = memmap.shape
    if labels is None:
        labels = np.arange(num_docs)
    similarity = encoder(queries)
    similarity = similarity.T / np.linalg.norm(similarity, axis=1)
    similarity = similarity.T.dot(memmap.T)
    ind = np.array([np.argpartition(s, -limit)[-limit:] for s in similarity])
    return labels[ind]


def find_labels(queries, mm, path=None, encoder=model.encode, limit=10, skip=1):
    """DUPLICATE of find_similar()"""
    path = Path(
        path or "/home/hobs/.nlpia2-data/wikipedia-20220228-titles-all-in-ns0.txt"
    )
    fin = path.open()
    header = fin.readline()
    if skip and not header.strip().lower() in ("title", "0"):
        fin.seek(0)
    labels = np.array([x.strip().replace("_", " ") for x in fin])

    similarity = encoder(queries)
    similarity = similarity.T / np.linalg.norm(similarity, axis=1)
    similarity = similarity.T.dot(mm.T)
    limit = 10
    ind = np.array([np.argpartition(s, -limit)[-limit:] for s in similarity])
    return labels[ind]
