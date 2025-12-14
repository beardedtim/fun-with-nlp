# Bible LM

## Creating Indexes

```python
process_bible_json(file_path)
```

This will expect a JSON file at `file_path` of shape

```json
{
  "Book Name": {
    "1": {
      "1": "verse text"
    }
  }
}
```

and will produce in `artifacts/indexes` files of `pre_Book Name.json` and `post_Book Name.json`.
Inside of `pre_` is the `text` of the given book and the `offsets` of shape `{ [verseNumber]: charOffset }`.
Inside of `post_` is the post-processed semantic triples and their count in the given book with the following
shape

```json
{
  "('s','v','o')": 2
}
```

This is the first step is getting the corpus ready for analytics.

## Creating Findings

### Lift

I want to understand how the occurance of a _semantic triple_ (_sub->pred->obj_) in a book of the Bible
relates to _other_ triples in the Bible. Specifically I wanted to understand the [lift](<https://en.wikipedia.org/wiki/Lift_(data_mining)>)
of these semantic triples.

```python
# Create a DataFrame from the triples created from the previous step
df = read_all_post_indexes(INDEX_DIR_PATH)
# Calculte the lift, grouped by `book`
df = calculate_lift_grouped(df, group_col='book')


# Only get triples that occour 3 or more times in a book
must_be_used_more_than_twice = df[df['count'] >= 3]

# Get the top 5 from each book by lift
top_5_df = must_be_used_more_than_twice.groupby('book', group_keys=False).apply(lambda x: x.nlargest(5, 'lift'))

# Save to image
# Save the top 5 by lift with only
save_df_triples(top_5_df, FINDINGS_DIR / "lift" / "top-5-lift.png")

# Save the top 5 by count
save_df_triples(
  df.groupby('book', group_keys=False).apply(lambda x: x.nlargest(5, 'count')),
  FINDINGS_DIR / "lift" / "top-5-count.png"
)


# Save for readying/later processing
book_output_data = {}

for bookName, bookDf in must_be_used_more_than_twice:
    # get the top 5 sorted by `lift` value
    top_5_lift = bookDf.sort_values(by='lift', ascending=False).head(5)
    # get the top 5 sorted by `count` value
    top_5_count = bookDf.sort_values(by='count', ascending=False).head(5)

    # save for later
    book_output_data[bookName] = {
        "by_count": top_5_count.to_dict(orient='records'),
        "by_lift": top_5_lift.to_dict(orient='records'),
    }

# write to the findings folder for later
write_json_file(FINDINGS_DIR / "lift" / "grouped_top_5.json", book_output_data)

```

## Saving into Neo4J for exploring

First, start docker

```sh
docker compose up -d
```

Next, save into graph in python

```python

df = read_all_post_indexes(INDEX_DIR_PATH)
try:
    save_into_graph(df)
finally:
    neo4jDriver.close()
```

Run these via the web ui exposed by neo

```
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX action_predicate IF NOT EXISTS FOR ()-[r:ACTION]-() ON (r.predicate);
```

to create some indexes. And here's some queries to run to explore more

```
# how often is something loved?
MATCH (s:Entity)-[r:ACTION]->(t:Entity)
WHERE toLower(r.predicate) = 'love'
RETURN s.name as Subject, r.predicate AS Predicate, t.name AS Object, r.book AS Book, sum(r.count) AS Frequency
ORDER BY Frequency DESC
```

```
MATCH (n:Entity)
// Find who n is connected to (undirected)
OPTIONAL MATCH (n)-[:ACTION]-(m:Entity)
WITH n, collect(distinct m.name) as Neighbors
RETURN
    n.name as Entity,
    size(Neighbors) as Degree,
    Neighbors[0..5] as SampleConnections
ORDER BY Degree DESC
LIMIT 20;
```
