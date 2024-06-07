import logging
import sys
from pathlib import Path

from transformers import pipeline

from knowt.llm import get_model_path, PROMPT_EXAMPLES, DEFAULT_LLM_MODEL, CLIENT
from knowt.search import VectorDB

try:
    log = logging.getLogger(__name__)
except NameError:
    log = logging.getLogger(f"{Path(__file__).name[:-3]}.__main__")

RAG_PROMPT_TEMPLATE = PROMPT_EXAMPLES[-1][-1]

RAG_SEARCH_LIMIT = 8
RAG_MIN_RELEVANCE = 0.3
RAG_N_SENTENCES = 2
RAG_TEMPERATURE = 0.05
RAG_SELF_HOSTED = True
RAG_MAX_NEW_TOKENS = 256


class RAG:

    def __init__(
        self,
        prompt_template=RAG_PROMPT_TEMPLATE,
        llm_model_name=DEFAULT_LLM_MODEL,
        search_limit=RAG_SEARCH_LIMIT,
        min_relevance=RAG_MIN_RELEVANCE,
        client=None,
        db=None,
        self_hosted=RAG_SELF_HOSTED,
        temperature=RAG_TEMPERATURE,
        max_new_tokens=RAG_MAX_NEW_TOKENS,
        n_sentences=RAG_N_SENTENCES,
        halucenate=False,
    ):
        self.pipe = None
        self.halucenate = halucenate or False
        self.temperature = temperature = min(temperature or 0, RAG_TEMPERATURE)
        self.max_new_tokens = max_new_tokens or RAG_MAX_NEW_TOKENS
        self.self_hosted = RAG_SELF_HOSTED if self_hosted is None else bool(self_hosted)
        self.client = client = client or CLIENT
        self.prompt_template = prompt_template
        self.llm_model_name = get_model_path(llm_model_name or DEFAULT_LLM_MODEL)
        self.hist = []
        self.search_limit = search_limit or RAG_SEARCH_LIMIT
        # TODO: these should be persisted in the db object
        self.min_relevance = int(min_relevance or 0) if min_relevance is not None else RAG_MIN_RELEVANCE
        self.n_sentences = int(n_sentences or 1) if min_relevance is not None else RAG_MIN_RELEVANCE

        if self_hosted:
            self.pipe = pipeline("text-generation", model=self.llm_model_name)

        if isinstance(db, VectorDB):
            self.db = db
        elif db is None:
            self.db = VectorDB(
                search_limit=self.search_limit, min_relevance=self.min_relevance
            )
        else:
            self.db = VectorDB(
                df=db, search_limit=self.search_limit, min_relevance=self.min_relevance
            )

    def setattrs(self, *args, **kwargs):
        if len(args) and isinstance(args[0], dict):
            kwargs.update(args[0])
        for k, v in kwargs.items():
            # TODO: try/except better here
            if not hasattr(self, k):
                log.error(
                    f'No such attribute "{k}" in a {self.__class__.__name__} class!'
                )
                raise AttributeError(
                    f"No such attribute in a {self.__class__.__name__} class!"
                )
            setattr(self, k, v)

    def ask(
        self,
        question,
        context=0,
        search_limit=None,
        min_relevance=None,
        n_sentences=None,
        prompt_template=None,
        **kwargs,
    ):
        """Ask the RAG a question, optionally reusing previously retrieved context strings

        Args:
          context (int|str): A str will be used directly in the LLM prompt.
            -1 => last context from history of chat queries
             0 => refresh the context using the VectorDB for semantic search
             1 => use the first context retrieved this session
             2 => use the 2nd context retrieved this session
        """
        self.question = question
        self.search_limit = search_limit = search_limit or self.search_limit
        self.min_relevance = min_relevance = int(min_relevance or 0) if min_relevance is not None else self.min_relevance
        self.n_sentences = n_sentences = int(n_sentences or 1) if n_sentences is not None else self.n_sentences
        self.prompt_template = prompt_template = prompt_template or self.prompt_template
        self.setattrs(kwargs)
        if (not context or context in [0, "refresh", "", None]) or not len(self.hist):
            topdocs = self.db.search(question, limit=search_limit, min_relevance=min_relevance, n_sentences=n_sentences)
            topdocs = topdocs[topdocs["relevance"] > min_relevance]
            context = "\n".join(list(topdocs[self.db.text_label]))
        if isinstance(context, int):
            try:
                context = self.hist[context]["context"]
            except IndexError:
                context = self.hist[-1]["context"]
        self.context = context = context or "Search returned 0 results."
        self.hist.append(
            {k: getattr(self, k) for k in "question context prompt_template".split()}
        )
        prompt = self.prompt_template.format(**self.hist[-1])  # **vars(self))
        self.hist[-1]['prompt'] = prompt
        results = self.run_model(prompt)
        self.hist[-1].update(results)  # answer=completion['content']
        for k, v in results.items():
            setattr(self, k, v)
        # TODO: function to flatten an openAI Completion object into a more open-standard interoperable format
        # FIXME: .hist rows should each be temporarily stored in a .turn dict with well-defined schema accepted by all functions
        return self.answer

    def run_model(self, prompt, self_hosted=None, max_new_tokens=None):
        """Generate text with self.pipe (HuggingFace) or openrouter from prompt (must already contain context or search results)"""
        self.max_new_tokens = max_new_tokens = max_new_tokens or self.max_new_tokens
        if self_hosted is not None:
            self.self_hosted = self_hosted
        if self.pipe:
            self.answer = self.pipe(prompt, max_length=None, max_new_tokens=max_new_tokens)[0][
                "generated_text"
            ]
            self.answers = [(self.answer, 0)]
            self.answer_id = f"self_hosted_{len(self.hist)+1}"
        else:
            self.completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://qary.ai",  # Optional, for including your app on openrouter.ai rankings.
                    "X-Title": "https://qary.ai",  # Optional. Shows in rankings on openrouter.ai.
                },
                model=self.llm_model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            self.answers = [
                (cc.message.content, cc.logprobs) for cc in self.completion.choices
            ]
            self.answer_id = self.completion.id
        self.answer, self.answer_logprob = self.answers[0]
        return dict(
            answers=self.answers,
            answer=self.answer,
            answer_id=self.answer_id,
            answer_logprob=self.answer_logprob,
        )


def main():
    question = " ".join(sys.argv[1:])
    # args = parser.parse_args(flags)
    if not len(question):
        question = input("AMA: ")
    main.rag = RAG() if main.rag is None else main.rag
    answers = main.rag.ask(question)
    print(answers)
    return(answers)
    # answers = ask_llm(
    #     question=question,
    #     model='auto',
    #     context='',
    #     prompt_template=PROMPT_NO_CONTEXT)


main.rag = None

if __name__ == "__main__":
    answers = main()
