from knowt.search import VectorDB


def test_vectordb():
    db = VectorDB()
    assert db.num_dim == 384
    assert db.embeddings.shape == (41531, 384)
