it = iter(generate_batches(iter(x.strip().replace('_', ' ') for x in fin), batch_size=1000))
for i, batch in tqdm(enumerate(it), total=nrows // batch_size):
    mm[i * batch_size:(i + 1) * batch_size, :] = model.encode(batch)
mm.flush()
    similarity = encoder(queries)
    similarity = similarity.T / np.linalg.norm(similarity, axis=1)
    similarity = similarity.T.dot(mm.T)


def find_similar(memmap, limit=1, queries=None, labels=None, encoder=model.encode):
    if isinstance(memmap, (str, Path)):
        memmap = np.memmap(memmap, mode='r')
    if isinstance(queries, str):
        queries = [queries]
    num_docs, num_dims = memmap.shape
    if labels is None:
        labels = np.arange(num_docs)
    similarity = encoder(queries)
    similarity /= np.linalg.norm(similarity, axis=1)
    similarity = similarity.dot(memmap.T)
    ind = np.array([
        np.argpartition(s, -limit)[-limit:]
        for s in similarity
    ])
    return labels[ind]


# find_similar(mm, limit=10, labels=list(fin))
fin.seek(0)
fin.readline()
labels = np.array(iter(x.strip().replace('_', ' ') for x in fin))
labels[0]
labels = np.array([x.strip().replace('_', ' ') for x in fin])
find_similar(mm, queries, limit=10, labels=labels)
queries = ["Kyiv", "Kyev"]
find_similar(mm, queries=queries, limit=10, labels=labels)
    similarity = encoder(queries)
    similarity = similarity.T / np.linalg.norm(similarity, axis=1)
    similarity = similarity.T.dot(mm.T)
encoder = model.encode
    similarity = encoder(queries)
    similarity = similarity.T / np.linalg.norm(similarity, axis=1)
    similarity = similarity.T.dot(mm.T)
    ind = np.array([
        ...: np.argpartition(s, -limit)[-limit:]
        ...: for s in similarity
        ...:])
limit = 10
    ind = np.array([
        ...: np.argpartition(s, -limit)[-limit:]
        ...: for s in similarity
        ...:])
ind
labels
labels[ind]
len(labels)
fout
mm.filename
fout = Path(_)
fout
path = fout.with_suffix('.labels.memmap')
mm2 = np.memmap('/home/hobs/.nlpia2-data/wikipedia-20220228-titles-all-in-ns0.labels.memmap', dtype=str, mode='w+', shape=(nrows, 1))
