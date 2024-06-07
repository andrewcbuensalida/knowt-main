from llm import *
from search import VectorDB
import pandas as pd

db = VectorDB()
# find the most trustworthy RAG model that won't claim to know anything about Python
answers = []
for model in MODELS:
    kwargs = dict(
        context='Python is fast?',
        question='What is Python?',
        model=model,
    )
    ans = rag(**kwargs)
    kwargs.update(dict(answer=ans[0]))
    answers.append(kwargs)
    df = pd.DataFrame(answers)
    print(df.iloc[-1])
    print()
pd.DataFrame(answers)
