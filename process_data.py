#!/usr/bin/env python
"""
  Script to process the notes by tokenizing them and merging the token:
  1. Load in the data
  2. Drop duplicates
  3. Merge `category`, `description`, and `text` into a new column called `note`
  4. Tokenize text using `scispacy` and create new column called `scispacy_note` to save tokenized text
  5. Save a csv file onto disk
"""
import pandas as pd
import spacy

from pathlib import Path

nlp = spacy.load('en_core_sci_md', disable=['parser', 'ner', 'tagger'])
raw_csv = Path('./data/raw_dataset.csv')
proc_csv = Path('./data/proc_dataset.csv')

def tokenize_text(text):
  tokens = [token.text for token in nlp(text)]
  return ' '.join(tokens)

def group_eth(eth):
  eth = eth.lower()
  if 'white' in eth:
    return 'white'
  elif 'black' in eth:
    return 'black'
  elif 'hispanic' in eth:
    return 'hispanic'
  elif 'asian' in eth:
    return 'asian'
  else:
    return 'unknown'

if __name__=='__main__':
  df = pd.read_csv(raw_csv)
  df.drop_duplicates(inplace=True)
  df['note'] = df['category'].str.cat(df['description'], sep='\n')
  df['note'] = df['note'].str.cat(df['text'], sep='\n')
  df['ethnicity'] = df['ethnicity'].apply(group_eth)
  df['processed_note'] = df['note'].apply(tokenize_text)
  df.drop(['text', 'description'], axis=1, inplace=True)
  df.to_csv(proc_csv, index=False)