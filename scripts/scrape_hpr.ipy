from search import VectorDB
from search import *
pd.read_csv('data/hpr_podcasts.csv')
df = _
for i, row in df.iter_rows():
    print(row.as_dict())
columns = 'seq_num title url host_name host_url full_title'.split()
columns += 'full_title subtitle series audio show_notes tags'.split()
len(df.columns)
columns
len(columns)
df.columns = columns
df.head()
df.iloc[0]
columns[6]
columns[6] = 'full_title_4digit'
df.iloc[0]
df.columns = columns
df.iloc[0]
for i, row in df.iter_rows():
    text = f"## {row['full_title_4digit']}\n\n {row['show_notes']}'
for i, row in df.iter_rows():
    text = f"## {row['full_title_4digit']}\n\n {row['show_notes']}"
    with open('data/hpr_corpus/{url[6:13].strip(/)}.txt') as fout:
        fout.write(text)
for i, row in df.iterrows():
    print(f'{row["url"][6:13].strip(/)}.txt')
    text = f"## {row['full_title_4digit']}\n\n {row['show_notes']}"
    with open(f'data/hpr_corpus/{row["url"][6:13].strip(/)}.txt') as fout:
        fout.write(text)
for i, row in df.iterrows():
    print(f'{row["url"][6:13].strip("/")}.txt')
    text = f"## {row['full_title_4digit']}\n\n {row['show_notes']}"
    with open(f'data/hpr_corpus/{row["url"][6:13].strip("/")}.txt') as fout:
        fout.write(text)
ls data/hpr_corpus
ls data/
for i, row in df.iterrows():
    print(f'{row["url"][6:13].strip("/")}.txt')
    text = f"## {row['full_title_4digit']}\n\n {row['show_notes']}"
    with open(f'data/corpus_hpr/{row["url"][6:13].strip("/")}.txt', 'w') as fout:
        fout.write(text)
db = VectorDB?
db = VectorDB(df='./data/corpus_hpr/hpr_sentences.csv')
db = VectorDB(df=Path('./data/corpus_hpr/hpr_sentences.csv'))
db = VectorDB()
db.cli()
%run llm
who
rag()
df.to_csv('data/test_llms.csv')
df
