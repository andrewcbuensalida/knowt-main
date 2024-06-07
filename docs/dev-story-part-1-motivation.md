# Part 1 - Motivation and demo

### 1. Motivation and demo

#### Goals:
- private
- teachable
- aligned not loyal (honest+compliant)
    - I do not want a loyal assistant that does everything I say.
    - Sometimes I get in a bad mood and have crazy ideas and I need an external tool to ground me in reality.
    - I want it to share my values for truth and the greater good.
- transparent (humble)
    - I want it to say "I don't know" or "I'm just a stochastic parot"
    - Defer to human-generated content that I have selected/curated.
- reliable and reproducible
    - I don't care about "hallucinations", that's what machines are programmed to do.
    - They exist purely on the Internet in the virtual world or dream world.
    - They do not feel pain or shame or even have any punishment when they do or say something stupid.
    - Their reward function is customer satisfaction... so they learn to flatter and brag and manipulate rather than how to be correct.

#### Architecture:
- LLMs
  - Essentially a Multilayer Markhov Model -- at each decision point you have a probability distribution that guides your next roll of the dice to chose the next word.
  - That's why experts like Timnit Gebru call them stochastic parrots, rather than AI.
  - The decision you are rolling the dice on for a language model is to predict the next word.
  - LLMs are just much much larger and more complicated markhov models than we've ever built before.
  - But at their core, transformers (LLM neural networks) predict the next word using a learned probability distribution.
    - Greedy search
    - Beam search
    - Receding horizon
- RAG
    - Retrieval
  - Generation
- Examples:
  - Maria Dyshel clued me into LlamaIndex 
    - At the March meetup SD Python User Group () Jason Koo  showed some basic use cases for retail customer service chatbots
    - Jason Koo (developer advocate for Neo4J) is trying to take RAG to the next level by incorporating knowledge graphs
    - At SD Python User Group he showed some basic use cases for retail customer service chatbots
    - It's unclear how easy it will be to automate the knowledge graph building part.
    - Greg Thompson's experiments with LlamaIndex revealed the importance of
      - Chunking
        - Sentence boundaries
        - Chunksize
        - Chunk-overlap
      - Extracting/generating metadata and prepending it to each chunk
        - (title, date, authorship)
    - 

#### References:
- [Knowt Project](https://gitlab.com/tangibleai/community/knowt)
- [NLPiA book](https://gitlab.com/hobs/nlpia-manuscript/)
- [LlamaIndex.ai](https://www.llamaindex.ai/)

## HPR Script

There are some questions that you would never trust with corporate "AI".
Do you take notes at school, work or ... life?
Ever wanted to know what you wrote down at your last exam?
What about the name of that cute person with the mishievious smile that you saw at San Diego Tech Coffee?
You probably have that info in a text file on your laptop somewhere, but you probably haven't ever used the word "mischievous" in the notes you jot down in a rush.
You may forget those details unless you have a tool like knowt to help you resurface them.
All you need to do is put your text notes into the "data/corpus" directory and knowt will take care of the rest.

Under the hood, Knowt implements a RAG (Retrieval Augmented Generative model).
So knowt first processes your private text files to create a searchable index of each passage of text you provide.
This gives it to perform "semantic search" on this indexed data blazingly fast, without using approximations.
See the project [final report](docs/Information Retrieval Systems.pdf) for more details.
To index a 10k documents should take less than a minute, and adding new documents takes seconds.
And answers to your questions take milliseconds.
Even if you wanted to ask some general question about some fact on Wikipedia, that would take less than a second (though indexing those 10M text stings took 3-4 hours on my two-yr-old laptop).
