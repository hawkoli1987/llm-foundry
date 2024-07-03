import json
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from streaming import StreamingDataset, Stream
import numpy as np
import random
import os

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
        print(f"peep into {lang}: {index_dict[lang][0]['zip_data']['basename']}")

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
def create_replay_index_3_lang(root_dir, out_dir, lang_ids, ratios):
    # to fix a random seed
    random.seed(17)
    index = []
    # mds_lang_path = [f'{path}/{lang}' for lang in lang_ids]

    
    for lang, ratio in zip(lang_ids, ratios):
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
        print(f"peep into {lang}: {shards[0]['zip_data']['basename']}")

        print(lang, 'has old shards size', len(shards))

        length = len(shards)
        n_replay = int(ratio * length)

        # randomly sample a 'ratio' (between 0 to 1) * 'length' nubmers of ids,
        # save the result into a list sampled_ids with len: int(ratio * length), value range: (0, length)
        if len(shards) >= n_replay:
            shards = random.sample(shards, n_replay)
            print(lang, f'has been sampled down to {len(shards)}')
        else:
            print(f"error in size estimation, n_replay {n_replay} should be less than len(shards) {len(shards)}")    
        
        # collect all lang shards into index
        index += shards

    # create a new subfolder under root_dir, name it as f'replay_{int(replay_ratio*100)}'
    replay_dir = os.path.join(root_dir, out_dir)
    os.makedirs(replay_dir, exist_ok=True)
    
    # save the index into a json file named 'index.json', under the new subfolder just created
    with open(os.path.join(replay_dir, 'index.json'), 'w') as f:
        json.dump({"shards": index, "version": 2}, f, indent=4)

    return index


# def merge_index_json_simple_approach(path, lang_ids):
#     infos = []
#     # mds_lang_path = [f'{path}/{lang}' for lang in lang_ids]

#     for lang in lang_ids:
#         lang_index_json = f'{path}/{lang}/index.json'
#         obj = json.load(open(lang_index_json))
#         for shard in range(len(obj['shards'])):
#             for which in ['raw_data', 'zip_data', 'raw_meta', 'zip_meta']:
#                 if obj['shards'][shard].get(which):
#                     basename = obj['shards'][shard][which]['basename']
#                     obj['shards'][shard][which]['basename'] = f"../gemma_tokenizer/{lang}/{basename}"
            
#         # only get half for rpv2_0_4 and en
#         if lang in ['rpv2_0_24', 'en']:
#             obj['shards'] = obj['shards'][:len(obj['shards'])//2]
            
#         infos += obj['shards']
#     print(infos[:20])

#     return infos

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


if __name__ == '__main__':
    # root_path = '/home/project/11003280/vault/ngan/SEA_fm_data/arf/data/50B_high_quality/gemma_tokenizer'
    # out_file = '/home/project/11003280/vault/ngan/SEA_fm_data/arf/data/50B_high_quality/gemma_tokenizer_simple_approach_index/index.json'
    
    # # # simple approach: half en, half rpv2_0_4, (zh, id, ms, vi, th) same 
    # # lang_ids = ['rpv2_0_24', 'en', 'zh', 'id', 'ms', 'vi', 'th']
    # # infos = merge_index_json_simple_approach(root_path, lang_ids)
    
    # # obj = {
    # #     'version': 2,
    # #     'shards': infos,
    # # }

    # # print(len(infos))
    # # with open(out_file, 'w') as out:
    # #     json.dump(obj, out)

    # # tokens_budget = 50e9 # 50B tokens
    # # domains_and_ratios = {
    # #     'rpv2_0_4': 1,
    # #     'rpv2_5_9': 0,
    # #     'en': 1,
    # #     'zh': 1,
    # #     'id': 1,
    # #     'ms': 1
    # # }
    
    # local_path = '/home/project/11003280/vault/ngan/SEA_fm_data/arf/data/50B_high_quality/gemma_tokenizer_simple_approach_index'
    # test_the_merge(local_path)

    # root_dir = '/fsx/yuli/cache/data/50B/with_replay'
    # lang_ids = ['rpv2_0_24', 'rpv2_25_49', 'en', 'zh', 'id', 'ms', 'vi', 'th']

    # # check_complete(root_dir, lang_ids)
    # for replay_ratio in [0.25, 0.5, 0.75, 1.0]:
    #     create_replay_index(replay_ratio, root_dir, lang_ids)
    
    # check_version(root_dir, ['replay_25','replay_50'])
    # replay_dir = os.path.join(root_dir, 'replay_25')
    # test_the_merge(replay_dir)



# (mosaicml_070_ngan) ubuntu@ip-10-0-8-190:/fsx/yuli$ /fsx/envs/mosaicml_070_ngan/bin/python /fsx/yuli/nscc_working/arf/mosaicml_workspace/scripts/replay.py

# en has old shards size 3554
# zh has old shards size 997
# id has old shards size 60
# ms has old shards size 44
# vi has old shards size 725
# th has old shards size 989

# en has new shards size = 2666 + 888 = 3554
# en has new shards size = 2666 + 888 = 3554
# en has new shards size = 1777 + 1777 = 3554
# en has new shards size = 889 + 2665 = 3554
# en has new shards size = 285 + 3269 = 3554

    # ###############################
    # root_dir = "/fsx/yuli/cache/data/50B/llama3"
    # lang_ids = ["en_pile", "id_hq", "id_pile1", "id_pile2", "ms_hq", "ms_pile"]
    # ratios = [0.5, 1, 1, 1, 1, 1]
    # out_dir='exp1_p1'
    # create_replay_index_3_lang(root_dir, out_dir, lang_ids, ratios)
    # # en_pile has old shards size 6435
    # # en_pile has been sampled down to 3217
    # # id_hq has old shards size 38
    # # id_pile1 has old shards size 655
    # # id_pile2 has old shards size 2006
    # # ms_hq has old shards size 2
    # # ms_pile has old shards size 132
    # ###############################

    ###############################
    root_dir = "/home/project/11003280/data/3lang/out_minicpm"
    lang_ids = ["en_pile", "id_hq", "id_pile1", "id_pile2", "ms_hq", "ms_pile"]
    ratios = [0.5, 1, 1, 1, 1, 1]
    out_dir='exp1_p1'
    create_replay_index_3_lang(root_dir, out_dir, lang_ids, ratios)
    # en_pile has old shards size 9161
    # en_pile has been sampled down to 4580
    # id_hq has old shards size 54
    # id_pile1 has old shards size 970
    # id_pile2 has old shards size 2517
    # ms_hq has old shards size 31
    # ms_pile has old shards size 184
    ###############################