import re

import bs4
import pandas as pd
import requests
from tqdm import tqdm
import numpy as np
from knowt.constants import DATA_DIR
from knowt.search import with_suffixes
import json
import yaml

HPR_CSV_PATH = DATA_DIR / "corpus_hpr" / "hpr_podcasts.csv.bz2"
BASE_DOMAIN = "hackerpublicradio.org"
BASE_URL_REGEX = r"\s*http[s]?:[/][/]*hackerpublicradio[.]org\s*"


def scrape_index(url=f"https://{BASE_DOMAIN}/eps/index.html"):
    """The HPR index page also contains T.W.A.T. (Today With A Techie) shows, but this ignores them"""
    resp = requests.get(url)
    bs = bs4.BeautifulSoup(resp.text, features="html.parser")
    episodes = [
        [
            ep["href"],
            ep.text,
            ep.next_sibling.next_sibling["href"],
            ep.next_sibling.next_sibling.text,
        ]
        for ep in bs.find_all("a")
        if ep.get("href", "").strip(".").strip("/").startswith("eps/hpr")
    ]
    df = pd.DataFrame(episodes, columns="url full_title user_url user".split())
    df = df.sort_values("url")
    df = df.reset_index(drop=True)
    df["seq_num"] = df.index.values + 1
    df["title"] = df["full_title"].str.split("::").str[-1]
    return df["seq_num title url user user_url full_title".split()]


def scrape_index_ignore_twat(url=f"https://{BASE_DOMAIN}/eps/index.html"):
    """Like scrape_index, but does more filtering of URLs and cleans the URLs before returning a DF"""
    resp = requests.get(url)
    bs = bs4.BeautifulSoup(resp.text, features="html.parser")
    episodes = [
        [
            ep["href"].strip(".").strip("/"),
            ep.text,
            (
                getattr(getattr(ep, "next_sibling", None), "next_sibling", None) or {}
            ).get("href", ""),
            getattr(ep.next_sibling.next_sibling, "text", ""),
        ]
        for ep in bs.find_all("a")
        if (
            ep
            and hasattr(ep, "next_sibling")
            and hasattr(ep.next_sibling, "next_sibling")
            and ep.get("href", "").strip(".").strip("/").startswith("eps/hpr")
        )
    ]
    episodes = [[parse_url(e[0]).get("url", "")] + e for e in episodes]

    # TODO: utilize the extracted hpr_num in the returned DF
    # episodes = [(list(e) + [un.get('url'), un.get('hpr_num'), un.get('episode_num')]
    #              if len(un) else (e, {}))
    #             for (e, un) in zip(episodes, url_nums)]

    df = pd.DataFrame(episodes, columns="url rawurl full_title user_url user".split())
    df = df.sort_values("url")
    df = df.reset_index(drop=True)
    df["seq_num"] = df.index.values + 1
    df["title"] = df["full_title"].str.split("::").str[-1]
    return df["seq_num title url user user_url full_title".split()]


def scrape_episode(
    url="./eps/hpr0030/", base_domain=BASE_DOMAIN, base_url_regex=BASE_URL_REGEX
):
    if url.lstrip(".").lstrip("/").startswith("eps/hpr"):
        url = "https://" + "/".join([base_domain, url.lstrip(".").lstrip("/")])
    resp = requests.get(url)
    s = bs4.BeautifulSoup(resp.text, features="html.parser")
    title, comments = s.find_all("h1")[1:3]
    subtitle, series = s.find_all("h3")[1:3]
    show_notes = list(series.next_siblings)[-1].next.next_sibling
    links = list(series.parent.find_all("a"))
    tags = [
        a.text
        for a in links
        if a.get("href", "").lstrip(".").lstrip("/").startswith("tags.html#")
    ]
    audio_urls = [
        a.get("href", "")
        for a in links
        if (a.text.lower().strip()[-3:] in "ogg spx mp3")
    ]
    row = dict(
        url=url.replace(base_url_regex, ".").lstrip(".").lstrip("/"),
        full_title_4digit=title.text,
        subtitle=subtitle.text,
        series=series.text,
        audio_urls=audio_urls,
        comments=comments,
        show_notes=show_notes.text,
        tags=yaml.dump_all([tags]).rstrip("\n"),
    )
    series.parent.find_all("a")
    return row


def clean_parse_urls(df, base_domain=BASE_DOMAIN, base_url_regex=BASE_URL_REGEX):
    df["url"] = df["url"].str.replace(
        base_url_regex, ".", regex=True
    )  # 'https://hackerpublicradio.org', '.')
    df["url"] = df["url"].str.replace(f"https://{base_domain}/", "")
    df["url"] = df["url"].str.lstrip(".").str.lstrip("/")
    df["hpr_num"] = df["url"].str.split("/").str[-2].str[3:].astype(int)
    df["episode_num"] = df["url"].apply(
        lambda u: int((re.findall(r"\b[hH][pP][rR](\d+)\b", u) or [-1])[0])
    )
    # df['episode_num'] = df.url.str.extract(r'\b[hH][pP][rR](\d+)\b').astype(int)
    return df


def parse_url(url):
    url = str(url).strip().strip("/")
    episode_num = int((re.findall(r"\b[hH][pP][rR](\d+)\b", url) or [-1])[0])
    hpr_num = int(url.split("/")[-2][3:])
    return dict(url=url, episode_num=episode_num, hpr_num=hpr_num)


def coerce_to_list(obj):
    if isinstance(obj, (tuple, set, np.ndarray)):
        return list(obj)
    if not isinstance(obj, (list, str)) and not obj:
        return []
    if isinstance(obj, str):
        obj = obj.strip().lstrip("[").rstrip("]").split(",")
        return [u.strip().strip('"').strip("'").strip() for u in obj]


def json_loads(obj):
    if obj is None:
        return np.nan
    if isinstance(obj, (tuple, set, np.ndarray)):
        return list(obj)
    if not isinstance(obj, (list, str)) and not obj:
        return []
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            # FIXME: need to also deal with Python syntax dicts (single-quoted keys or values)
            if obj[0] in "[({":
                return coerce_to_list(obj)
    return str(obj)


def str2list(s, fillna="[]"):
    if isinstance(fillna, str):
        fillna = yaml.safe_load(fillna)
    if pd.isna(s) or isinstance(s, None):
        return fillna
    if not isinstance(s, str):
        return s
    return yaml.safe_load(s)


def clean(df, base_url_regex=BASE_URL_REGEX):
    """Drop duplicates, make urls relative, convert `tags` str2list, create `title_tags`, `hpr_num`"""

    # create episode_num, hpr_num columns from url after stripping base_url from url
    df = clean_parse_urls(df, base_url_regex=BASE_URL_REGEX)

    # dflatest doesn't have 'tags' ??!!  Will it AFTER scrape_episode on all the urls?
    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("[]")
        # Pandas saves lists as str(list) which yaml.safe_load() recognizes but json doesn't
        df["tags"].apply(yaml.safe_load)
        df["title_tags"] = (
            df["episode_num"].astype(str) + ": " + df["title"].str.strip()
        )
        df["title_tags"] += " (" + df["tags"].apply(yaml.safe_load).str.join(", ") + ")"
    df = df.sort_values("episode_num")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values("episode_num")  # Redundant?
    df = df.reset_index(drop=True)
    return df
    # TODO: validate rows and drop invalid ones
    # df = df.drop(labels=4039)


def normalize_url(url, host_regex=BASE_URL_REGEX, normalized_host=""):
    """Use regex to strip hostname and scheme to create relative URL, for deduplication

    >>> url = f'https://hackerpublicradio.org/eps/hpr4055/index.html'
    >>> normalize_url(url)
    'eps/hpr4055/index.html'
    >>> url = ' http://hackerpublicradio.org/eps/hpr4055/index.html '
    >>> normalize_url(url)
    'eps/hpr4055/index.html'
    """
    return (
        re.sub(pattern=host_regex, repl=normalized_host, string=url)
        .strip()
        .strip(".")
        .strip("/")
    )


def update(update_existing=False, base_domain=BASE_DOMAIN, host_regex=BASE_URL_REGEX):
    """Download the latest episode index from HackerPublicRadio.org and add new episodes to HPR_CSV_PATH"""
    dfold = pd.read_csv(HPR_CSV_PATH).sort_values("url")
    dfold = clean(dfold)
    dflatest = scrape_index().sort_values("url")
    dflatest = clean_parse_urls(dflatest)
    scraped_urls = sorted(dfold["url"].str.strip().unique())
    episodes = []
    for i, row in tqdm(dflatest.iterrows()):
        row = row.to_dict()
        urls = row.get("audio_urls", "")
        if not isinstance(urls, list):
            urls = coerce_to_list(urls)
        url = normalize_url(row["url"], "").strip().strip(".").strip("/")
        if url in scraped_urls:
            continue
        row.update(url=scrape_episode(row["url"], ""))
        episodes.append(row)
        scraped_urls += episodes[-1]["url"].strip()

    latest_csv_path = with_suffixes(HPR_CSV_PATH, ".latest.csv")
    dflatest.to_csv(latest_csv_path)
    dflatest = clean(dflatest)
    if len(episodes) == len(dflatest):
        dflatest = dflatest.set_index("url", drop=False).to_dict()
        for i, episode in tqdm(episodes):
            dflatest[episode["url"]].update(episode)
        dflatest = pd.DataFrame(dflatest)
    else:
        dflatest = pd.concat([dfold, pd.DataFrame(episodes)], axis=0)
    dflatest.to_csv(HPR_CSV_PATH, index=None)
    return dflatest


if __name__ == "__main__":
    df = update()
