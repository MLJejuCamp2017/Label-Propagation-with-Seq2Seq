import numpy as np
import unicodedata
import os
import pickle

def split_data(src_path, tgt_path, src_lang='ko', tgt_lang='en'):
    """
    Split parallel dataset into training/validation/test dataset.
    Half of train dataset is gonna be monolingual dataset. 

    Especially, there are some heuristics to clean dataset:
        1. character-level separation of sentences.
        2. restriction of number of characters.

    It'll store splitted dataset under "data/raw/" directory. 
    """
    # Make directory if it doesn't exist
    directory = './data/raw'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Read all parallele dataset into lists
    with open(src_path, 'r') as f_src, open(tgt_path, 'r') as f_tgt:
        sources_raw = f_src.readlines()
        targets_raw = f_tgt.readlines()

        ### rule 1. character-level separation of sentences 
        sources_raw = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in sources_raw]
        targets_raw = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in targets_raw]
        ###

    # Convert characters into integers
    all_bytes, sources, targets = [], [], []
    for i in range(len(sources_raw)):
        src = [ord(ch) for ch in sources_raw[i]]
        tgt = [ord(ch) for ch in targets_raw[i]]
        sources.append(src)
        targets.append(tgt)
        all_bytes.extend(src+tgt)
        
    ### rule 2. remove obsolete characters & sentences
    unique_bytes = np.unique(all_bytes)

    for i in sources:
        i.sort()
    for i in targets:
        i.sort()

    banned_char_ids = [0, 61, 96, 97, 98, 101, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 173, 212, 216]
    banned_char_ids.extend(list(range(255, len(unique_bytes))))
    banned_char_list = unique_bytes[banned_char_ids]
    banned_char_list.sort()

    def searchsorted(x, y):
        idx_x, idx_y = 0, 0
        n_x, n_y = len(x), len(y)
        while(idx_x<n_x and idx_y<n_y):
            if x[idx_x] < y[idx_y]:
                idx_x += 1
            elif x[idx_x] > y[idx_y]:
                idx_y += 1
            else:
                return True
        return False

    banned_ss_ids = []
    for i, s in enumerate(sources):
        if searchsorted(s, banned_char_list):
            banned_ss_ids.append(i)
        print(i, end='\r')

    for i in banned_ss_ids[::-1]:
        sources_raw.pop(i)
        targets_raw.pop(i)
    ###

    ss_1 = [unicodedata.normalize('NFC', x) for x in sources_raw]
    ss_2 = [unicodedata.normalize('NFC', x) for x in targets_raw]

    # Shuffle dataset
    N = len(ss_1)
    ids = np.arange(N)
    np.random.shuffle(ids)

    with open('%s/%s.total' % (directory, src_lang), 'w') as f_src,\
            open('%s/%s.total' % (directory, tgt_lang), 'w') as f_tgt :
        start = 0
        end = len(ss_1)
        for i in range(start, end):
            f_src.write(ss_1[ids[i]]+'\n')
            f_tgt.write(ss_2[ids[i]]+'\n')

    with open('%s/%s.train' % (directory, src_lang), 'w') as f_src,\
            open('%s/%s.train' % (directory, tgt_lang), 'w') as f_tgt:
        start = 0
        end = start + (N // 10) * 4
        for i in range(start, end):
            f_src.write(ss_1[ids[i]]+'\n')
            f_tgt.write(ss_2[ids[i]]+'\n')

    with open('%s/%s.train.mono' % (directory, src_lang), 'w') as f_src,\
                open('%s/%s.train.mono' % (directory, tgt_lang), 'w') as f_tgt:
            start = (N // 10) * 4
            end = start + (N // 10) * 4
            for i in range(start, end):
                f_src.write(ss_1[ids[i]]+'\n')
                f_tgt.write(ss_2[ids[i]]+'\n')

    with open('%s/%s.valid' % (directory, src_lang), 'w') as f_src,\
                open('%s/%s.valid' % (directory, tgt_lang), 'w') as f_tgt:
            start = (N // 10) * 8
            end = start + (N // 10) * 1
            for i in range(start, end):
                f_src.write(ss_1[ids[i]]+'\n')
                f_tgt.write(ss_2[ids[i]]+'\n')

    with open('%s/%s.test' % (directory, src_lang), 'w') as f_src,\
                open('%s/%s.test' % (directory, tgt_lang), 'w') as f_tgt:
            start = (N // 10) * 9
            end = N
            for i in range(start, end):
                f_src.write(ss_1[ids[i]]+'\n')
                f_tgt.write(ss_2[ids[i]]+'\n')


def make_voca(src_lang='ko', tgt_lang='en'):
    # Make directory if it doesn't exist
    directory = './data/cache'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # load ko-en parallel corpus
    with open('./data/raw/%s.total' % src_lang, 'r') as f:
        sources_raw = f.readlines()
        sources_raw = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in sources_raw]

    with open('./data/raw/%s.total' % tgt_lang, 'r') as f:
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

    voca_path = directory + '/preload_voca.pickle'

    # make vocabulary
    unique_all_bytes = list(np.unique(all_bytes))
    unique_all_bytes.sort()
    index2byte = [0, 1] + unique_all_bytes  # add <EMP>, <EOS>
    byte2index = {}
    for i, b in enumerate(index2byte):
        byte2index[b] = i
    voca_size = len(index2byte)

    with open(voca_path, 'wb') as f:
        pickle.dump([byte2index, index2byte, voca_size], f)

if __name__ == "__main__":
    split_data('./data/raw/crawl_dict_ko.txt', 
            './data/raw/crawl_dict_en.txt', 'ko', 'en')
    make_voca('ko', 'en')
