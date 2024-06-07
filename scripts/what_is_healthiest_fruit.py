from llm import *
from search import *

db = VectorDB()
db.search("What is the healthiest fruit?")
context = "\n".join([s for s in contextdf["sentence"]])
rag("What is the healthiest fruit?", context=context)
q = "How much exercise is healthiest?"
context = "\n".join([s for s in df2["sentence"]])
rag(q, context=context)
