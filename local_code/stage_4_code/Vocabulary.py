
'''
Maps words to integer IDs (built from training data only) and encodes token lists
into fixed-length, left-padded ID sequences.
'''

from collections import Counter

class Vocabulary:
    PAD, UNK = 0, 1

    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']

    def build(self, token_lists):
        counts = Counter()
        for tokens in token_lists:
            counts.update(tokens)
        for word, count in counts.most_common():
            if count >= self.min_freq:
                 new_id = len(self.idx2word)
                 self.word2idx[word] = new_id
                 self.idx2word.append(word)
    
    def __len__(self):
        return len(self.idx2word)
    
    def encode(self, tokens, max_len):
        ids = [self.word2idx.get(w, self.UNK) for w in tokens]
        ids = ids[:max_len]
        ids = [self.PAD] * (max_len - len(ids)) + ids

        return ids
    
    def decode(self, ids):
        return [self.idx2word[i] for i in ids]