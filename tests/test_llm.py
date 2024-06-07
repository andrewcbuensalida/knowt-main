from knowt.llm import RAG
from knowt.search import VectorDB
from transformers.pipelines import TextGenerationPipeline


def test_rag():
    db = VectorDB('corpus_hpr', max_files=100)
    rag = RAG(db, self_hosted=False, encoder='spacy')
    assert rag.db.num_dim == 384
    assert rag.db.embeddings.shape == (41531, 384)
    assert isinstance(rag.pipe, TextGenerationPipeline)
