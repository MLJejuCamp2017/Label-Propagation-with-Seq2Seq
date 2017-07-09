import numpy as np
import tensorflow as tf
import unicodedata
import pickle
from glob import glob
import os

class KOEN(object):
    """
    Load Korean to English Parallel Corpus
    """
    
    def __init__(self, batch_size=32, mode='train'):
        self.src_lang = 'ko'
        self.tgt_lang = 'en'

        if not self._restore(mode):
            # load train corpus
            self.source, self.target = self._load_corpus(mode=mode)
            self.num_data = len(self.source)
            self._save(mode)

        # calc total batch count
        self.num_batch = self.num_data // batch_size
        self.batch_size = batch_size

        # print info
        tf.logging.info('Train data loaded.(total data=%d, total batch=%d)' % (self.num_data, self.num_batch))

    def _save(self, mode):
        # Make directory if it doesn't exist
        directory = './data/cache'
        if not os.path.exists(directory):
            os.makedirs(directory)
                    
        with open('data/cache/%s2%s.%s.pickle' % (self.src_lang, self.tgt_lang, mode), 'wb') as f:
            pickle.dump([self.byte2index, self.index2byte, self.voca_size,
                self.max_len, self.num_data, self.source, self.target], f)

    def _restore(self, mode):
        if glob('data/cache/%s2%s.%s.pickle' % (self.src_lang, self.tgt_lang, mode)):
            with open('data/cache/%s2%s.%s.pickle' % (self.src_lang, self.tgt_lang, mode), 'rb') as f:
                [self.byte2index, self.index2byte, self.voca_size,
                        self.max_len, self.num_data,
                        self.source, self.target] = pickle.load(f)
            return True
        else:
            return False


    def _load_corpus(self, mode='train'):

        # load ko-en parallel corpus
        with open('data/raw/%s.%s' % (self.src_lang, mode), 'r') as f:
            sources_raw = f.readlines()
            sources_raw = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in sources_raw]

        with open('data/raw/%s.%s' % (self.tgt_lang, mode), 'r') as f:
            targets_raw = f.readlines()
            targets_raw = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in targets_raw]

        # make character-level parallel corpus
        all_bytes, sources, targets = [], [], []
        for i in range(len(sources_raw)):
            src = [ord(ch) for ch in sources_raw[i]]
            tgt = [ord(ch) for ch in targets_raw[i]]
            sources.append(src)
            targets.append(tgt)
            all_bytes.extend(src+tgt)

        voca_path = 'data/cache/preload_voca.pickle'
        with open(voca_path, 'rb') as f:
            [self.byte2index, self.index2byte, self.voca_size] = pickle.load(f)

        self.max_len = 150

        # remove short and long sentence
        src, tgt = [], []
        for s, t in zip(sources, targets):
            if 0 <= len(s) < self.max_len and 0 <= len(t) < self.max_len:
                try:
                    [self.byte2index[ch] for ch in s]
                    [self.byte2index[ch] for ch in t]
                except KeyError as e:
                    continue

                src.append(s)
                tgt.append(t)


        # convert to index list and add <EOS> to end of sentence
        for i in range(len(src)):
            src[i] = [self.byte2index[ch] for ch in src[i]] + [1]
            tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1]

        # zero-padding
        for i in range(len(tgt)):
            src[i] += [0] * (self.max_len - len(src[i]))
            tgt[i] += [0] * (self.max_len - len(tgt[i]))

        return src, tgt

    def to_batch(self, sentences):
        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i]] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        return sentences

    def print_index(self, indices, sysout=True):
        ret = []
        for i, index in enumerate(indices):
            str_ = ''
            for j, ch in enumerate(index):
                if ch > 1:
                    str_ += chr(self.index2byte[ch])
                elif ch == 1:  # <EOS>
                    break
            if sysout:
                print(unicodedata.normalize('NFC', str_))
            else:
                ret.append(unicodedata.normalize('NFC', str_))
        if not sysout:        
            return ret