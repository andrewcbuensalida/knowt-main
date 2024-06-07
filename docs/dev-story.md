# Knowt Series of HPT Episodes

#### Outline
1. Motivation and demo
  - Goals:
    - private
    - teachable
    - aligned (honest+compliant)
    - transparent (humble)
    - reliable (no "halucenation")
    - loyal
  - gitlab.com/tangibleai/community/knowt
  - NLPiA release date? 
2. Scraping data
  - OpenAI ruined it for everybody
    - reddit
    - GitHub (GitLab has always been tricky)
    - StackOverflow
    - news sites
    - Twitter
  - LLMs have poluted the infosphere (internet search with "before:2023")
  - `wget` recursively
  - `requests.get`
  - `pandas` and `bz2` dataframe for DB 
  - `pandas.read_html`
  - `bs4` to extract title, subtitle, hrefs (show notes, transcripts) 
  - `re`
  - new T.W.A.T. episodes (anyone want to help)
  - robustness ( `getattr`, `dict.get`, `if obj or` )
3. Doctests
  - DDD + TDD
  - `doctest.testmod`
  - `pytest`
  - `gitlab-ci.yml` (GitHub actions not open standard)
4. Text search
  - Inverted index (wikipedia.org/wiki/Inverted_index)
  - data science feature extraction and vectorization
  - count vectors are hyperdimensional, can only efficiently index/find discrete, sparse
  - `sklearn.feature_extraction.text.CountVectorizer`
  - `sklearn.feature_extraction.text.TfidfVectorizer`
5. SpaCy
  - `spacy.cli.download`
  - `nlp = spacy.load`
  - `for tok in nlp(text)`
  - `nlp(text).sents`
  - `nlp(text).vector` custom 1000D Word2Vec vectors better for toxic comments
6. Transformers
  - Language model
  - `transformers.pipeline`
  - `np.dot()`
7. Generation
  - openrouter.com
  - "auto" router
  - mystral-7B
  - prompt template
  - prompt engineering and "AI" training
8. Python packaging: Poetry vs setuptools
  - poetry documentation incomplete and opinionated
  - dependency sometimes breaks other scientific computing apps like jupyter-console/ipython
  - some features not possible in Poetry
  - `--editable` (only recently added after years of resistance)
  - knowt = knowt.search:main (already a linux search app)
  - ask = knowt.search:main 
9. Future work
  - knowt.nlpia.org or qary.ai
  - video.nlpia.org
  - meet.nlpia.org
  - social.nlpia.org or social.qary.ai
  - test set and metric reporting
  - quality test automation (pre-commit? post-push? post-commit?)
  - hyperparameter table of test results
  - index raw text documents for start and end of sentences
  - search for paragraphs or sentence-gram range
  - build from seed sentences to gather up a few neighbors
  - delete redundant search results
  - embed sentence pairs and compare to averaged pair of vectors for relevance in test set
  - vary number of sentences provided for each search result
  - accumulate relevance scores to determine number of sentences instead of thresholding
  - highlight search results in ASCII color
  - search ranking algorithm reinforcement learning
  - teaching it new "facts"
  - recording notes
  - searching wikipedia
  - writing code
  - llm fall-back
  - RAG on your code

