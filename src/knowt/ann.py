""" Approximate Nearest Neighbor search and indexing """

from py_lsh import LSHash
import numpy as np


def generate_hashes(embeddings, hash_size=None, input_dims=None, num_hashtables=None):
    input_dims = int(input_dims or len(embeddings[0]))
    input_dims = int(input_dims)
    hash_size = int(hash_size or int(2 * np.sqrt(input_dims)) + 2)
    print(
        f"hash_size: {hash_size}\ninput_dims: {input_dims}\nnum_hashtables: {num_hashtables}\n"
    )

    num_hashtables = num_hashtables or 1
    lsh = LSHash(
        hash_size=hash_size, input_dim=input_dims, num_hashtables=num_hashtables
    )
    for e in embeddings:
        lsh.index(e)
    return lsh
