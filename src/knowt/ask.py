# ask.py
""" Ask questions (designed to be called from a command-line script)
1. retrieve text using semantic search and a VectorDB('corpus/') database
2. ask questions of an llm.RAG(db=VectorDB('corpus/'))
3. ask questions of an LLM directly (e.g. using openrouter "auto" or Mistral)
4. TBD: generate code using Mistral-13B
5. TBD: generate code with a RAG('Mistral-13B', db=corpus_gitlab or corpus_readthedocs)

TODO:
0. persist conversation turns to disk (.joblib? .jsonl.bz2? .csv.bz2)
1. Persist VectorDB to joblib?
2. Persist VectorDB to np.memmap
3. Incrementally add to np.memmap
4. Allow user to upvote/downvote answers from commandline

"""

from knowt.llm import CLIENT, get_model, RAG_PROMPT_TEMPLATE, RAG
from knowt.search import VectorDB


PROMPT_TEMPLATE_ASK_LLM = (
    "You are an AI virtual assistant and search engine."
    "You answer questions truthfully and succinctly as if you are a witness at a legal trial.\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER: "
)


def ask_llm(
    question="What is Python?",
    prompt_template=PROMPT_TEMPLATE_ASK_LLM,
    client=CLIENT,
    model="meta-llama/llama-2-13b-chat",
    **kwargs,
):
    """Ask a pure LLM (on OpenRouter) a question, without RAG or other enhancements"""
    # FIXME: create a small VectorDB containing model names/descriptions for intent matching
    model = get_model(model)
    prompt = prompt_template.format(question=question, **kwargs)
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://qary.ai",  # Optional, for including your app on openrouter.ai rankings.
            "X-Title": "https://qary.ai",  # Optional. Shows in rankings on openrouter.ai.
        },
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return [c.message.content for c in completion.choices]


def ask_rag(
    question="What is Python?",
    prompt_template=RAG_PROMPT_TEMPLATE,
    client=CLIENT,
    model="mistralai/mistral-7b-instruct",
    **kwargs,
):
    """Ask a RAG (semantic search for sentences + OpenRouter for Mystral) a question

    alias knowt='python -c "from knowt.llm import ask_rag; print(ask_rag($@))"'
    """
    # FIXME: create persistent daemon/service containing the RAG model
    rag = RAG(prompt_template=prompt_template, llm_model=model, client=client)
    context = rag.db.search_pretty(question)
    print(context)
    # print("FIXME: context above doesn't seem to be used in the rag.ask() or doesnt have enough relevant info!")
    answers = rag.ask(question, context=context)
    return answers


def ask_search(question="What is Python?", **kwargs):
    """Ask a pure LLM (on OpenRouter) a question, without RAG or other enhancements"""
    # FIXME: create persistent daemon/service containing the RAG model
    # rag = RAG(prompt_template=prompt_template, llm_model=model, client=client)
    db = VectorDB(**kwargs)
    answers = db.search_pretty(question)
    return answers
