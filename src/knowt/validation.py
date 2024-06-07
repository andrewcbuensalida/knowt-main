# evaluation.py
import pandas as pd
from pathlib import Path
from constants import EXAMPLES_PATH

from knowt.search import VectorDB, TEXT_LABEL


def add_example(query, df_examples=None, db=None, limit=10, text_label=TEXT_LABEL):
    if df_examples is None:
        df_examples = add_example.df_examples or EXAMPLES_PATH
    if isinstance(df_examples, str):
        df_examples = Path(df_examples)
    if isinstance(df_examples, Path):
        if df_examples.is_file():
            df_examples = pd.read_csv(df_examples)
        else:
            df_examples = pd.DataFrame()
            df_examples.to_csv(df_examples)
    top_docs = db.search(query)
    print(top_docs)
    print("Vote for the top 4 search results.")
    print(
        "For each relevant passage, type its row number and hit [Enter] for each relevant document."
    )
    print("Start with the best (most relevant) document and work your way down.")
    print(
        "If fewer than 4 of the search results are relevant, then leave the answer empty to quit voting."
    )
    for i in range(4):
        human_ranks = [(int(input("Rank 1 document: ")), i + 1) for i in range(4)]

    new_rows = []
    if top_docs.empty:
        print("No documents found for this query.")
        new_rows = [dict(query=query, sentence="", rank=1, index=None, filename=None)]
    for i, (index, row) in enumerate(top_docs.iterrows()):
        row_dict = row.astype(dict)
        row_dict["query"] = query
        row_dict["index"] = index
        # row_dict['rank'] = i + 1
        new_rows.append(row_dict)
        row_dict["human_rank"] = None
        row_dict["human_similarity"] = None
        new_rows.append(row_dict)
    for i, (index, row) in enumerate(top_docs.iterrows()):
        print(f"{i+1} ({row['relevance']:.3f}): {row[db.text_label]}")

    for i, rank in human_ranks:
        new_rows[i - 1]["human_rank"] = i + 1  # 1-offset

    new_rows_df = pd.DataFrame(new_rows)
    test_set = pd.concat([df_examples, new_rows_df], ignore_index=True)

    return test_set


add_example.df_examples = None


if __name__ == "__main__":
    db = VectorDB()
    while True:
        query = input("Enter search query ([ENTER] or 'exit' to quit): ")
        if query.lower().strip() in ("exit", "exit()", ""):
            break
        db.search_pretty(query, limit=5)
