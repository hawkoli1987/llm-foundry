import json
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from streaming import StreamingDataset, Stream
import numpy as np
import random
import os
import math
from torch.utils.data import DataLoader
import streaming
import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def check_complete(root_dir, lang_ids):
    for lang in lang_ids:
        lang_index_json = f'{root_dir}/{lang}/index.json'
        with open(lang_index_json) as f:
            obj = json.load(f)
        shards = obj['shards']
        for shard in shards:
            if 'zip_data' in shard:
                basename = shard['zip_data']['basename']
                if not os.path.exists(f"{root_dir}/{lang}/{basename}"):
                    print(f'missing {basename}')
    print('checking completed')
    return

def check_version(root_dir, lang_ids):
    for lang in lang_ids:
        cnt = 5

        lang_index_json = f'{root_dir}/{lang}/index.json'
        with open(lang_index_json) as f:
            obj = json.load(f)
        shards = obj['shards']
        for shard in shards:
            if shard['version'] != 2:
                print('error in:', shard['zip_data']['basename'])
            else:
                if cnt > 0:
                    print(lang, type(shard['version']))
                    cnt -= 1
    return

# Replace the 'en' with 'rpv2', given a replay_ratio, and all the lang_ids as one list, 
# save it as 'replay_<ratio>' under root_dir, which only contain the index.json
def create_replay_index(replay_ratio, root_dir, lang_ids):
    # to fix a random seed
    random.seed(17)
    index_dict = {}
    index = []
    index_rp = []
    # mds_lang_path = [f'{path}/{lang}' for lang in lang_ids]

    # change relative path of each shard to absolute
    for lang in lang_ids:
        lang_index_json = f'{root_dir}/{lang}/index.json'
        with open(lang_index_json) as f:
            obj = json.load(f)
        shards = obj['shards']
        
        for shard in shards:
            for which in ['raw_data', 'zip_data', 'raw_meta', 'zip_meta']:
                if which not in shard:
                    continue
                basename = shard[which]['basename']
                shard[which]['basename'] = f"{root_dir}/{lang}/{basename}"
        index_dict[lang] = shards
        # print(f"peek into {lang}: {index_dict[lang][0]['zip_data']['basename']}")

        # collect the rp into a flatten list, served as alternative sample pool
        if lang in ['rpv2_0_24', 'rpv2_25_49']:
            index_rp += index_dict[lang]

    # replace en with rp by sampling
    for lang in [lang for lang in lang_ids if lang not in ['rpv2_0_24', 'rpv2_25_49']]:
        
        shards = index_dict[lang]
        print(lang, 'has old shards size', len(shards))

        # for en, replace a portion by rp
        if lang == 'en':
            length = len(shards)
            n_replay = int(replay_ratio * length)

            # randomly sample a 'replay_ratio' (between 0 to 1) * 'length' nubmers of ids,
            # save the result into a list sampled_ids with len: int(replay_ratio * length), value range: (0, length)
            if len(index_rp) > n_replay:
                shards_new = random.sample(index_rp, n_replay)
                shards_old = random.sample(shards, length - n_replay)
            else:
                shards_new = index_rp
                shards_old = random.sample(shards, length - len(index_rp))
            shards = shards_new + shards_old
            print(lang, f'has new shards size = {len(shards_old)} + {len(shards_new)} = {len(shards)}')
        
        # collect all lang shards into index
        index += shards

    # create a new subfolder under root_dir, name it as f'replay_{int(replay_ratio*100)}'
    replay_dir = os.path.join(root_dir, f'replay_{int(replay_ratio*100)}')
    os.makedirs(replay_dir, exist_ok=True)
    
    # save the index into a json file named 'index.json', under the new subfolder just created
    with open(os.path.join(replay_dir, 'index.json'), 'w') as f:
        json.dump({"shards": index, "version": 2}, f, indent=4)

    return index

# root_dir: /fsx/yuli/cache/data/50B/llama3
# lang_ids: ["en_pile", "id_hq", "id_pile1", "id_pile2", "ms_hq", "ms_pile"]
# The en_pile_ratio will be the required en_pile size / total en_pile size (65B)
# save it as 'replay_<ratio>' under root_dir, which only contain the index.json
def create_replay_index_langs(root_dir, seq_len, out_dir, target_token_sizes_each_lang: dict, shuffle=True):
    # to fix a random seed
    random.seed(17)
    index = []
    # mds_lang_path = [f'{path}/{lang}' for lang in lang_ids]

    tokens_source_dict = calculate_total_tokens_each_subset(root_dir, seq_len)

    ratio_dict = {}

    for lang, total_size in tokens_source_dict.items():
        if lang in target_token_sizes_each_lang:
            ratio_dict[lang] = target_token_sizes_each_lang[lang] / total_size
            print(f"{lang} will sample {target_token_sizes_each_lang[lang]:.1f} B Tokens from {total_size:.1f} B Tokens >> sampling ratio {ratio_dict[lang]:.2f}")    

    for lang, ratio in ratio_dict.items():
        lang_index_json = f'{root_dir}/{lang}/index.json'
        with open(lang_index_json) as f:
            obj = json.load(f)
        shards = obj['shards']
        
        # change relative path of each shard to absolute
        for shard in shards:
            for which in ['raw_data', 'zip_data', 'raw_meta', 'zip_meta']:
                if which not in shard:
                    continue
                basename = shard[which]['basename']
                shard[which]['basename'] = f"{root_dir}/{lang}/{basename}"
        # print(f"peep into {lang}: {shards[0]['zip_data']['basename']}")

        length = len(shards)
        n_replay = int(ratio * length)

        # randomly sample a 'ratio' (between 0 to 1) * 'length' nubmers of ids,
        # save the result into a list sampled_ids with len: int(ratio * length), value range: (0, length)
        if len(shards) >= n_replay:
            shards_ = random.sample(shards, n_replay)
            print(lang, f'has been sampled down from {len(shards)} to {len(shards_)}')
            shards = shards_
        elif ratio > 1:
            # legitimate upsampling            
            int_part = math.floor(ratio)
            decimal_part = ratio - int_part
            shards_ = shards*int_part + random.sample(shards, int(decimal_part * length))
            print(lang, f'has been sampled up from {len(shards)} to {len(shards_)}')
            shards = shards_
            
        else:
            print(f"error in size estimation, n_replay {n_replay} should be less than len(shards) {len(shards)}")    
        
        # collect all lang shards into index
        index += shards

    if shuffle:
        random.shuffle(index)

    # create a new subfolder under root_dir, name it as f'replay_{int(replay_ratio*100)}'
    replay_dir = os.path.join(root_dir, out_dir)
    os.makedirs(replay_dir, exist_ok=True)

    # save the index into a json file named 'index.json', under the new subfolder just created
    with open(os.path.join(replay_dir, 'index.json'), 'w') as f:
        json.dump({"shards": index, "version": 2}, f, indent=4)

    return index


def calculate_total_tokens_each_subset(path, seq_len):

    dirs = os.listdir(path)
    tokens_source_dict = {}

    # print(path)
    for d in dirs:
        d_path = f'{path}/{d}/index.json'
        with open(d_path, 'r') as f:
            data = json.load(f)['shards']
        for shard in data:
            num_sample = shard['samples']
            if d not in tokens_source_dict:
                tokens_source_dict[d] = 0
            tokens_source_dict[d] += num_sample*seq_len/10**9
            
    tokens_source_dict = dict(sorted(tokens_source_dict.items()))
    print(path)
    for k, v in tokens_source_dict.items():
        print(k, f"{v:.2f}")

    return tokens_source_dict

def calculate_total_tokens_in_merged(root_dir:str, seq_len):

    path = os.path.join(root_dir, "index.json")
    
    tokens_source_dict = {}
    with open(path, 'r') as f:
        data = json.load(f)['shards']
    for shard in data:
        num_sample = shard['samples']
        shard_name = shard['raw_data']['basename'].split('/')[-2]
        if shard_name not in tokens_source_dict:
            tokens_source_dict[shard_name] = 0
        tokens_source_dict[shard_name] += num_sample *seq_len / 10**9
    tokens_source_dict = dict(sorted(tokens_source_dict.items()))
    
    total_size = 0
    print(path)
    for k, v in tokens_source_dict.items():
        print(k, f'{v:.1f}')
        total_size += v
    print(f'total size is: {total_size:.2f}')

def test_the_merge(replay_dir):
    tokenizer_path = '/fsx/yuli/cache/model/gemma2b'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = StreamingDataset(local=replay_dir, shuffle=False, keep_zip=True)

    total_bs = len(dataset)
    print(total_bs)
    # idx = 0
    # for idx in range(total_bs):
    #     if idx >= 16011235:
    #         ids = np.frombuffer(dataset[int(f'{idx}')]['tokens'], dtype=np.int64).copy().tolist()
    #         tokens = tokenizer.convert_ids_to_tokens(ids)
    #         # print(ids)
    #         print(tokens)
    #         print('=========')
    #         break

def test_num_tokens():
    tokenizer_path = '/fsx/yuli/cache/model/llama3_tokenizer'
    # tokenizer_path = 'lib/gemma-2b'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    while True:
        text_input = input('Input:')
        if not text_input:
            continue
        
        ids = tokenizer.encode(text_input)
        print(len(text_input), len(ids), len(text_input)/len(ids))

if __name__ == '__main__':

    ############################### llama3 10-90
    root_dir = "/fsx/yuli/cache/data/out_hero_llama3"
    seq_len = 8192
    target_token_sizes_each_lang = {
        "en_pile": 0,
        "en_hq": 4,
        "id_pile1": 10.08,
        "id_hq": 1.20,
        "th_pile1": 5,
        "th_pile2": 6.28,
        "th_hq": 0.72,
        "vi_pile": 10.76,
        "vi_hq": 1.24
    }
    out_dir='replay10'
    
    create_replay_index_langs(root_dir, seq_len, out_dir, target_token_sizes_each_lang, shuffle=True)
    calculate_total_tokens_in_merged(os.path.join(root_dir, out_dir), seq_len)
