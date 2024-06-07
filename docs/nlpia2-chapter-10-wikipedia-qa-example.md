=== Wikipedia for the ambitious reader

If training your model on the text in this book seems a little constraining for you, consider going "all in" and training your model on Wikipedia.
After all, Wikipedia contains all of the human knowledge, at least the knowledge that the _wisdom of the crowd_ (humanity) thinks is important.

Be careful though.
You will need a lot of RAM, disk space, and compute throughput (CPU) to store, index and process the 60 million articles on Wikipedia.
And you will need to deal with some insidious quirks that could corrupt your search results invisibly.
And it's hard to curate billions of words of natural language text.

If you use full-text search on PyPi.org for "Wikipedia" you won't notice that "It's A Trap!"footnote:[Know Your Meme article for "It's A Trap" (https://knowyourmeme.com/memes/its-a-trap)]
You might fall into the trap with `pip install wikipedia`.
Don't do that.
Unfortunately, the package called `wikipedia` is abandonware, or perhaps even intentional name-squatting malware.
If you use the `wikipedia` package you will likely create bad source text for your API (and your mind):

[source,console]
----
$ pip install wikipedia
----

[source,python]
----
>>> import nlpia2_wikipedia.wikipedia as wiki
>>> wiki.page("AI")
DisambiguationError                       Traceback (most recent call last)
...
DisambiguationError: "xi" may refer to:
Xi (alternate reality game)
Devil Dice
Xi (letter)
Latin digraph
Xi (surname)
Xi Jinping
----

That's fishy.
No NLP preprocessor should ever corrupt your "AI" query by replacing it with the capitalized proper name "Xi".
That name is for a person at the head of one of the most powerful censorship and propaganda (brainwashing) armies on the planet.
And this is exactly the kind of insidious spell-checker attack that dictatorships and corporations use to manipulate you.footnote:[(https://theintercept.com/2018/08/01/google-china-search-engine-censorship/)]
To do our part in combating fake news we forked the `wikipedia` package to create `nlpia2_wikipedia`.
We fixed it so you can have a truly open source and honest alternative.
And you can contribute your own enhancements or improvements to pay it forward yourself.

You can see here how the `nlpia2_wikipedia` package on PyPi will give you straight answers to your queries about AI.footnote:["It Takes a Village to Combat a Fake News Army" by Zachary J. McDowell & Matthew A Vetter (https://journals.sagepub.com/doi/pdf/10.1177/2056305120937309)]

[source,console]
----
$ pip install nlpia2_wikipedia
----

[source,python]
----
>>> import nlpia2_wikipedia.wikipedia as wiki
>>> page = wiki.page('AI')
>>> page.title
'Artificial intelligence'
>>> print(page.content)
Artificial intelligence (AI) is intelligence—perceiving, synthesizing,
and inferring information—demonstrated by machines, as opposed to
intelligence displayed by non-human animals or by humans.
Example tasks ...
>>> wiki.search('AI')
['Artificial intelligence',
 'Ai',
 'OpenAI',
...
----

Now you can use Wikipedia's full-text search API to feed your retrieval-augmented AI with everything that humans understand.
And even if powerful people are trying to hide the truth from you, there are likely a lot of others in your "village" that have contributed to Wikipedia in your language.

----
>>> wiki.set_lang('zh')
>>> wiki.search('AI')
['AI',
 'AI-14',
 'AI-222',
 'AI＊少女',
 'AI爱情故事',
...
----

Now you know how to retrieve a corpus of documents about any topic that is important to you.
If it's not already, AI and large language models will certainly be important to you in the coming years.
You can teach your retrieval augmented question answering system from the previous section to answer questions from any knowledge you can find on the internet, including Wikipedia articles about AI.
You no longer have to rely on search engine corporations to protect your privacy or provide you with factual answers to your questions.
You can build your own retrieval-augmented LLMs to answer questions factually for you and those you care about at your workplace or in your community.