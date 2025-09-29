"""
Given a built split of LibriPhrase, make the following files:
- ${LibriSpeech_split}_sentences.csv
    - Columns: sentence_index,file,speaker,text
- ${LibriPhrase_split}_keywords.csv
    - Columns: word_index,file,speaker,text,class
    - Words are union of all unique anchors and comparisons from LibriPhrase

"""

import os
import pandas as pd
from tqdm import tqdm
from typing import *
from praatio import textgrid

tqdm.pandas()

# input files
DATASETS = os.environ['DATASETS']

LIBRISPEECH_ROOT = os.path.join(DATASETS, "LibriSpeech")
LIBRISPEECH_SPLIT = "train-clean-100"
LIBRISPEECH_AUDIO = os.path.join(LIBRISPEECH_ROOT, LIBRISPEECH_SPLIT)

LIBRIPHRASE_ROOT = os.path.join(DATASETS, 'LibriPhrase')
LIBRIPHRASE_DATA = os.path.join(LIBRIPHRASE_ROOT, 'data')
LIBRIPHRASE_SPLIT = "libriphrase_diffspk_all"
LIBRIPHRASE_AUDIO = os.path.join(LIBRIPHRASE_DATA, LIBRISPEECH_SPLIT)

# output files
SENTENCES_CSV = os.path.join(LIBRIPHRASE_DATA, f"{LIBRISPEECH_SPLIT}_sentences.csv")
WORDS_CSV = os.path.join(LIBRIPHRASE_DATA, f"{LIBRIPHRASE_SPLIT}_keywords.csv")

def get_librispeech_path(relpath: str) -> str:
    """
    Arguments:
        relpath:    String of relative path to LibriSpeech datafile
    Returns:
        str of absolute path to LibriSpeech datafile
    """
    return os.path.join(LIBRISPEECH_ROOT, relpath)

def get_libriphrase_audio_path(relpath: str) -> str:
    """
    Arguments:
        relpath:    String of relative path to LibriPhrase audio file
    Returns:
        str of absolute path to LibriPhrase audio file
    """
    return os.path.join(LIBRIPHRASE_DATA, relpath)

def textgrid_to_df(textgrid_path: str) -> pd.DataFrame:
    """
    Arguments:
        textgrid_path:  Absolute path to textgrid file to open.
    Returns:
        textgrid_df:    Pandas DataFrame containing data for textgrid.
    """
    textgrid_obj = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)
    rows = []
    for tier in textgrid_obj.tiers:
        for start, end, value in tier.entries:
            if not value:
                continue
            rows.append({'start': start, 'end': end, 'value': value, 'tier': tier.name})
    textgrid_df = pd.DataFrame(rows)
    return textgrid_df

def get_librispeech_textgrid(librispeech_relpath: str) -> pd.DataFrame:
    """
    Arguments:
        librispeech_relpath:    String of relative path to LibriSpeech datafile.
                                Can have any or no file extension.
    Returns:
        textgrid_df:            Pandas DataFrame containing data for textgrid corresponding
                                to LibriSpeech datafile.
    """
    librispeech_relpath = os.path.splitext(librispeech_relpath)[0]
    librispeech_abspath = get_librispeech_path(librispeech_relpath)
    textgrid_path = librispeech_abspath + '.TextGrid'
    return textgrid_to_df(textgrid_path)
    

def get_unique_keywords(keyword_df: Optional[pd.DataFrame]=None) -> List[str]:
    if keyword_df is None:
        keyword_df = pd.read_csv(WORDS_CSV)
    return list(keyword_df['keyword'].unique())

def get_keyword_tokens(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None
    ) -> pd.DataFrame:
    """
    Arguments:
        keyword:    String representing keyword/phrase to query
        keyword_df: Optional Pandas DataFrame containing keyword data.
                    If not passed, load from $WORDS_CSV file.
    Returns:
        pandas.DataFrame, subset of `keyword_df` corresponding to the given keyword
    """
    if keyword_df is None:
        keyword_df = pd.read_csv(WORDS_CSV)
    keyword_mask = keyword_df['keyword']==keyword
    return keyword_df[keyword_mask]

def get_random_keyword_token(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None,
        num_records: int=1,
    ) -> pd.Series:
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        keyword_df:     Optional Pandas DataFrame containing keyword data.
                        If not passed, load from $WORDS_CSV file.
        num_records:    Number of rows to sample, default 1.
    Returns:
        pandas.Series, randomly sampled row of `keyword_df`
        corresponding to the given keyword
    """
    keyword_tokens = get_keyword_tokens(keyword, keyword_df)
    rows = keyword_tokens.sample(num_records)
    if num_records == 1:
        return rows.iloc[0]
    return rows
    
def get_keyword_sentences(
        keyword: str,
        sentence_df: Optional[pd.DataFrame]=None,
    ) -> pd.DataFrame:
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        sentence_df:    Optional Pandas DataFrame containing sentence data.
                        If not passed, load from $SENTENCES_CSV file.
    Returns:
        pandas.DataFrame, subset of `sentence_df` containing to the given keyword
    """
    if sentence_df is None:
        sentence_df = pd.read_csv(SENTENCES_CSV)
    keyword_mask = sentence_df['sentence'].str.contains(keyword)
    return sentence_df[keyword_mask]

def get_random_keyword_sentence_pair(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None,
        sentence_df: Optional[pd.DataFrame]=None,
        same_source: bool=False,
        same_speaker: bool=False,
    ) -> Tuple[pd.Series, pd.Series]:
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        keyword_df:     Optional Pandas DataFrame containing keyword data.
                        If not passed, load from $WORDS_CSV file.
        sentence_df:    Optional Pandas DataFrame containing sentence data.
                        If not passed, load from $SENTENCES_CSV file.
        same_source:    bool indicating whether the keyword token should
                        originate from the same audio as the sentence.
        same_speaker:   bool indicating whether the keyword token should
                        originate from the same speaker as the sentence.

    Returns:
        (keyword_row, sentence_row), randomly sampled row of `keyword_df`
        corresponding to the given keyword
    """
    keyword_row = get_random_keyword_token(keyword, keyword_df)
    keyword_sentences = get_keyword_sentences(keyword, sentence_df)
    source_mask = keyword_sentences['file']==keyword_row['sentence_file']
    speaker_mask = keyword_sentences['speaker']==keyword_row['speaker']
    if same_source:
        return keyword_row, keyword_sentences[source_mask].iloc[0]
    if same_speaker:
        return keyword_row, keyword_sentences[speaker_mask].sample().iloc[0]
    return keyword_row, keyword_sentences[~speaker_mask].sample().iloc[0]

def get_random_negative_keyphrase(
        keyword: str,
        sentence_df: Optional[pd.DataFrame]=None,
        speaker: Optional[str]=None,
        same_speaker: bool=False,
        num_records: int=1,
    ):
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        sentence_df:    Optional Pandas DataFrame containing sentence data.
                        If not passed, load from $SENTENCES_CSV file.
        speaker:        
        same_speaker:   bool indicating whether the keyword token should
                        originate from the same speaker as the sentence.
        num_records:    Number of rows to sample, default 1.

    Returns:
        keyword_row, randomly sampled row of `keyword_df`
        that does not contain the specified keyword
    """
    keyword_mask = sentence_df['sentence'].str.contains(keyword)
    negative_sentences = sentence_df[~keyword_mask]
    if speaker and same_speaker:
        speaker_mask = sentence_df['speaker']==speaker
        rows = negative_sentences[speaker_mask].sample(num_records)
    elif speaker:
        rows = negative_sentences[~speaker_mask].sample(num_records)
    else:
        rows = negative_sentences.sample(num_records)
        
    if num_records==1:
        return rows.iloc[0]
    return rows