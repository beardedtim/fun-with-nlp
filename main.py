import json
from pathlib import Path
import spacy
from collections import Counter
from rdflib import Graph, URIRef, Literal, Namespace
import ast

ARTIFACTS_DIR = Path('./artifacts')
RAW_DIR = ARTIFACTS_DIR / 'raw'
BIBLE_DIR = RAW_DIR / 'bible'
INDEX_DIR = ARTIFACTS_DIR / 'indexes'

nlp = spacy.load("en_core_web_trf")

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

def main():
    print("Hello from bible-lm!")
    
    # save data to indexes
    # file_path = BIBLE_DIR / 'niv.json'
    # process_bible_json(file_path=file_path)


if __name__ == "__main__":
    main()
