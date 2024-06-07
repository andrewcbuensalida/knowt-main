def update_with_key_abbreviations(d, seps=list(' -_'), positions=[0, 1, -1]):
    """ Add keys to hugging face and openrouter model name mapping by abbreviating and lowercasing full model paths

    >>> update_key_abbreviations({'hello-world/bye-world': 'hello-world/bye-world'}, {'hello-world/bye-planet': 'hello-world/bye-planet'})

    """
    d = dict(d or {})
    d2 = dict()
    for v in list(d.values()):
        if v not in d:
            d[v] = v
    for k in d:
        for sep in seps:
            for i in positions:
                for newk in str(k).split("/"):
                    for fold in str, str.lower:
                        newk = fold(newk)
                        if sep:
                            newks = newk.split(sep)
                        else:
                            newks = [newks]
                        if -len(newks) <= i < len(newks):
                            newk = newks[i].lower()
                            if newk not in d and newk not in d2:
                                d2[newk] = d[k]
    d.update(d2)
    return d
