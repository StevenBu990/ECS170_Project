
'''
Text cleaning for Stage 4 (required by the dataset README:
remove stop words and punctuation, normalize words).
Reviews and jokes are cleaned differently because the tasks differ.
'''

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# negations flip sentiment ("not good"), so they must NOT be removed
NEGATIONS = {'not', 'no', 'nor', 'never', 'nothing', 'none', 'cannot', 'neither', 'nobody', 'nowhere'}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATIONS

def clean_review(text):
    '''Movie review -> list of words for sentiment classification.'''
    text = text.lower()
    text = re.sub(r"n't", ' not', text)

    text = re.sub(r"<[^>]+>", ' ', text)
    text = re.sub(r"[^a-z\s]", ' ', text)

    words = text.split()

    words = [w for w in words if w not in STOP_WORDS]

    return words


def clean_joke(text):
    '''Joke -> list of tokens for generation. Keeps stop words and ? .  ! , as their own tokens,
    because the model has to generate complete sentences.'''
    text = text.lower()
    text = re.sub(r'([?.!,])', r' \1 ', text)
    text = re.sub(r"[^a-z0-9'?.!,\s]", ' ', text)
    return text.split()