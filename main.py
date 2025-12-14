import json
from pathlib import Path
import spacy
from collections import Counter
from rdflib import Graph, URIRef, Literal, Namespace
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import urllib.parse
from neo4j import GraphDatabase

ARTIFACTS_DIR = Path('./artifacts')
RAW_DIR = ARTIFACTS_DIR / 'raw'
BIBLE_DIR = RAW_DIR / 'bible'
INDEX_DIR = ARTIFACTS_DIR / 'indexes'
FINDINGS_DIR = ARTIFACTS_DIR / "findings"
INDEX_DIR_PATH = Path(INDEX_DIR)

nlp = spacy.load("en_core_web_trf")
uri = "bolt://localhost:7687"
username = "neo4j"
password = "ou812sosueme"
neo4jDriver = GraphDatabase.driver(uri, auth=(username, password))

# RDF Triple Info
RDF_Graph = Graph()
NAMESPACE_BASE = "https://mck-p.com/bible"

def read_json_file(path):
    """
    Given a path: string, returns a Dict
    of the JSON value
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File Not Found")
        raise
    except json.JSONDecodeError:
        print(f"JSON Decode Error")
        raise
    except Exception as e:
        print(f"Uknown Error Reading JSON File: {e}")
        raise

def write_json_file(path, data):
    """
    Given path: string and data, saves to disk
    """
    try:
        with open(path, "w", encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving {e}")
        
        raise

#
# These could all be the same function but I want to be able to
# test my steps by using chapter, then book, then bible instead of
# just running it against the full text of the bible right away
#
def concat_chapter_text(chapter, initial_offset = 0):
    """
    Given a Dict of { [versNum]: verseText }, returns the full text along with
    each verse's start offset
    """
    result = {
        'text': '',
        'offsets': {}
    }

    for verseNum, verseText in chapter.items():
        result['offsets'][verseNum] = len(result['text']) + initial_offset
        result['text'] += verseText + ' '

    return result

def concat_book_text(book, initial_offset = 0):
    """
    Given a Dict of { [chapterNum]: Chapter }, returns the full text along with
    each chapter's verses as offsets
    """
    result = {
        'text': '',
        'offsets': {}
    }

    for chapterNum, chapter in book.items():
        chapter_info = concat_chapter_text(chapter, len(result['text']) + initial_offset)
        result['offsets'][chapterNum] = chapter_info['offsets']
        result['text'] += chapter_info['text'] + '\n\n'
    
    return result
        
def concat_bible(bible):
    """
    Given a Dict of { [bookNames]: Book }, returns the full text along with
    each book's chapter's verses as offsets
    """
    result = {
        "text": '',
        'offsets': {}
    }

    for bookName, book in bible.items():
        book_info = concat_book_text(book, len(result['text']))
        result['offsets'][bookName] = book_info['offsets']
        result['text'] += book_info['text'] + '\n\n\n\n\n'
    
    return result


def generate_doc_from_text(text):
    """
    Generates a single Spacy doc for all other transformations
    """
    doc = nlp(text)

    return doc

def process_text(text):
    """
    Given a string, will return a Counter of term and frequency
    """
    tokens = []
    triples = []
    doc = generate_doc_from_text(text)

    for token in doc:
        # process all tokens for freq
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num:
            tokens.append(token.lemma_.lower())
        # if this is the main verb
        if token.pos_ == "VERB":
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ == "nsubj":
                    subj = child.lemma_.lower()
                if child.dep_ in ("dobj", "pobj"):
                    obj = child.lemma_.lower()
            
            if subj and obj:
                triples.append((subj, token.lemma_.lower(), obj))

    return {
        'freq': Counter(tokens),
        'triples': Counter(triples),
    }

def add_to_graph(data, prefix):
    """
    Given the JSON {
     "('sub','verb','obj')": <count>
    }

    insert into the graph the s,v,o triples
    """
    namespace = Namespace(prefix)
    for key, _ in data.items():
        s_raw, p_raw, o_raw = ast.literal_eval(key)

        s, p, o = namespace[s_raw], namespace[p_raw], namespace[o_raw]

        RDF_Graph.add((s, p, o))

def make_uri(term, base="http://mck-p.com/bible/data/"):
    """
    Turns the given term and base into a namespaced term
    so it can be used as a URI
    """
    safe_term = urllib.parse.quote(str(term))
    return f"<{base}{safe_term}>"

def make_book_uri(book_name):
    safe_book = urllib.parse.quote(str(book_name))
    return f"<http://mck-p.com/bible/book/{safe_book}>"

def create_nquad(df):
    return (
        "<< " + df['s_uri'] + " " + df['p_uri'] + " " + df['o_uri'] + " >> " +
        "<http://mck-p.com/bible/ontology/count>" + " " +
        '"' + df['count'].astype(str) + '"^^<http://www.w3.org/2001/XMLSchema#integer> ' +
        df['g_uri'] + " ."
    )

def configure_uri_for_df(df):
    df['s_uri'] = df['source'].apply(lambda x: make_uri(x))
    df['p_uri'] = df['edge'].apply(lambda x: make_uri(x))
    df['o_uri'] = df['target'].apply(lambda x: make_uri(x))
    df['g_uri'] = df['book'].apply(make_book_uri)
    df['nquad'] = create_nquad(df)

    return df
    

def parse_triples_from_key(str):
    """
    Given a string of "('a', 'b', 'c')", returns a tuple of ('a', 'b', 'c')
    """
    clean_key = str.strip("()")
    [a, b, c] = [p.strip(" '\"") for p in clean_key.split(", ")]

    return (a, b, c)

def df_from_triples(triples):
    """
    Given an object of { <triple in str form>: count }
    returns a pandas dataframe of the triples and their count
    """
    rows = []
    for triple_str, count in triples.items():
        (source, edge, target) = parse_triples_from_key(triple_str)
        rows.append({
            'source': source,
            'edge': edge,
            'target': target,
            'count': int(count)
        })
    
    return pd.DataFrame(rows)

def calculate_lift(df):
    """
    Given a dataframe of { source, edge, target, count }
    return the lift
    """
    # how many total { source, edge, target } instances do we have?
    total = df['count'].sum()

    # Support(Object)
    obj_count = df.groupby('target')['count'].sum()
    obj_prob = obj_count / total

    # Support(Subject, Predicate) :: Antecedent
    antecedent_counts = df.groupby(['source', 'edge'])['count'].sum()
    antecedent_prob = antecedent_counts / total

    # Join Probability P(s, p, o)
    df['joint_prob'] = df['count'] / total

    # No idea tbh
    df['prob_consequent'] = df['target'].map(obj_prob)
    df = df.join(antecedent_prob.rename('prob_antecedent'), on=['source', 'edge'])

    # Metrics we care about
    #
    # Confidence measures the conditional probability of the object, given the subject and predicate. 
    # It answers the question: 
    #           "When the subject $s$ is involved in relation $p$, how often is the target $o$?"
    #
    #
    df['confidence'] = df['joint_prob'] / df['prob_antecedent']

    # Lift is the most critical metric for semantic insight. It compares the observed confidence of the 
    # rule to the expected confidence if the antecedent and consequent were independent. It corrects for
    # the baseline popularity of the object.
    #
    # Lift = 1: 
    #   The entities are statistically independent. The co-occurrence is coincidental.
    #
    # Lift > 1: 
    #   The entities are positively correlated. The subject and predicate actively "lift" the probability
    #   of the object occurring.
    #
    # Lift < 1: The entities are negatively correlated. The presence of the subject makes the object less
    #   likely than its baseline probability
    #
    df['lift'] = df['confidence'] / df['prob_consequent']

    return df

def calculate_lift_grouped(df, group_col=None):
    """
    Calculates lift either globally or relative to a specific group (e.g., 'book').
    
    Args:
        df: The dataframe of triples {source, edge, target, count}
        group_col: (Optional) The column name to group by (e.g., 'book')
    """
    # 1. Determine the "Universe" size
    # If grouping, total is the word count per book. If not, it's the whole corpus.
    if group_col:
        total = df.groupby(group_col)['count'].transform('sum')
    else:
        total = df['count'].sum()

    # 2. Support(Object) :: P(o)
    # How often does this target appear in this specific universe?
    if group_col:
        # Group by Book AND Target to get local object counts
        obj_counts = df.groupby([group_col, 'target'])['count'].transform('sum')
    else:
        # Map global counts to the rows
        obj_counts = df.groupby('target')['count'].transform('sum')
        
    df['prob_consequent'] = obj_counts / total

    # 3. Support(Subject, Predicate) :: P(s, p)
    # How often does this pair appear in this specific universe?
    if group_col:
        ant_counts = df.groupby([group_col, 'source', 'edge'])['count'].transform('sum')
    else:
        ant_counts = df.groupby(['source', 'edge'])['count'].transform('sum')

    df['prob_antecedent'] = ant_counts / total

    # 4. Joint Probability :: P(s, p, o)
    df['joint_prob'] = df['count'] / total

    # 5. Calculate Metrics
    # Confidence = P(o | s,p)
    df['confidence'] = df['joint_prob'] / df['prob_antecedent']

    # Lift = P(s,p,o) / ( P(s,p) * P(o) )
    df['lift'] = df['confidence'] / df['prob_consequent']

    return df

def process_bible_json(file_path):
    """
    Given the path to a json of {[bookName]: Book}, save to indexes the processed
    results
    """
    print(f"File: {file_path}")
    data = read_json_file(file_path)

    for book, bookData in data.items():
        print(f"I am processing {book}\n")
        book_pre_processed = concat_book_text(bookData)
        print(f"First ~1k: {book_pre_processed['text'][:1000]}\n\n")
        print(f'Saving to indexes')
        file_name = "pre_" + book + ".json"

        index_file_path = INDEX_DIR / file_name

        print(f"Writing to {index_file_path}")

        write_json_file(index_file_path, book_pre_processed)
        print(f"Wrote to disk\n\n")

        book_post_processed = process_text(book_pre_processed['text'])
        
        print(f"Book Finished {book}\n")
        print(f"10 Most Common\n\n{book_post_processed['triples'].most_common(10)}")

        file_name = "post_" + book + ".json"

        index_file_path = INDEX_DIR / file_name

        print(f"Saving to indexes")
        print(f"Writing to {index_file_path}")
        data_to_save = {}
    
        for key_tuple, count in book_post_processed['triples'].items():
            # The key conversion is the crucial step: 
            # Convert the tuple key (e.g., ('term1', 'tagA')) into a string
            # The str() function works perfectly for this.
            string_key = str(key_tuple) 
            
            data_to_save[string_key] = count

        write_json_file(index_file_path, data_to_save)

        print(f"Wrote to disk\n\n")

def generate_scatter_plot(df):
    """
    Given a Dataframe that has had lift calculated,
    show a scatter plot
    """
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(
        df['count'], 
        df['confidence'], 
        c=df['lift'], 
        cmap='viridis', 
        s=100, 
        alpha=0.7
    )
    plt.colorbar(scatter, label='Lift')
    plt.title('Rule Metrics: Support vs Confidence (Color by Lift)')
    plt.xlabel('Count (Support)')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.5)
    # Optional: Label interesting points
    for i, row in df.iterrows():
        label = f"{row['source']}->{row['target']}"
        plt.annotate(label, (row['count'], row['confidence']), fontsize=8)
            
    plt.tight_layout()
    plt.savefig('scatter_plot.png') 
    plt.close() # CHANGE: Close the plot to free memory
    print("Saved scatter_plot.png")

def read_all_post_indexes(index_dir):
    """
    Given the content dir that holds all of the 
    pre_<book>.json and post_<book>.json files,
    this returns a single dataframe
    """
    file_glob = "post_*.json"
    file_paths_iterable = index_dir.glob(file_glob)
    all_dfs = []
    for filename in file_paths_iterable:
        print(f"Parsing {filename}")
        triples = read_json_file(filename)
        df_book = df_from_triples(triples)
        book_name = os.path.basename(filename).replace('.json','').replace('post_', '')
        df_book['book'] = book_name
        print(f"Parsed")
        print(f"Describe: {df_book.describe()}")
        print(f"Info: {df_book.info()}")
        all_dfs.append(df_book)
    
    df = pd.concat(all_dfs, ignore_index=True)

    return df

def save_df_triples(df, file_path):
    """
    Given a Dataframe with the column { book, lift, triple, count }, prints
    it
    """
    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col="book", col_wrap=3, sharey=False, sharex=False, height=4, aspect=1.5)
    g.map_dataframe(sns.barplot, x="lift", y="triple", palette="viridis")
    # 1. Add borders (spines) to every subplot
    for ax in g.axes.flat:
        # Turn on all 4 borders
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True) 
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        
        # Optional: Make them slightly thicker or a specific color
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
    # Formatting
    g.set_titles("{col_name}")
    g.set_axis_labels("Lift", "Triple")
    g.fig.suptitle('Top 5 Triples per Book')
    plt.subplots_adjust(top=0.9) # Add space for title
    # Save
    plt.savefig(file_path, bbox_inches='tight')


def save_into_graph(df, batch_size=10000):
    """
    Saves the dataframe triples into Neo4J in <batch_size>
    """
    df_clean = df.fillna(value={"count": 0, "book": "Unknown"})
    data_list = df_clean.to_dict('records')

    total = len(data_list)
    print(f"----Starting Ingest of {total} rows-----")
    query = """
    UNWIND $rows AS row
    MERGE (s:Entity {name: row.source})
    MERGE (t:Entity {name: row.target})
    CREATE (s)-[:ACTION {
        predicate: row.edge,
        count: row.count,
        book: row.book
    }]->(t)
    """

    with neo4jDriver.session() as session:
        for i in range(0, total, batch_size):
            print(f'-----Now at {i}-----')
            batch = data_list[i : i + batch_size]
            session.run(query, rows=batch)
            
            print(f'-----Done with {i} through {min(i+batch_size, total)}-----')

def main():
    print("Hello from bible-lm!")
    
    # save data to indexes
    # file_path = BIBLE_DIR / 'niv.json'
    # process_bible_json(file_path=file_path)

    # read data from indexes
    # file_path = INDEX_DIR / 'post_Genesis.json'
    # post_processed_genesis = read_json_file(file_path)
    # df = df_from_triples(post_processed_genesis)

    # total_obs = df['count'].sum()

    # print(f'Total observered {total_obs}')
    # df = calculate_lift(df)
    # print(f"Describe: {df.describe()}")
    # print(f"Info: {df.info()}")
    # generate_scatter_plot(df)

    # index whole bible, then compare books
    df = read_all_post_indexes(INDEX_DIR_PATH)
    try:
        save_into_graph(df)
    finally:
        neo4jDriver.close()

    # df = configure_uri_for_df(df)
    # df['nquad'].to_csv(ARTIFACTS_DIR / "dataframe" / "bible_data.trig", index=False, header=False, quoting=3, escapechar=None)

    

if __name__ == "__main__":
    main()
