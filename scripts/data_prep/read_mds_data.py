
import numpy as np
from streaming import StreamingDataset, Stream
from torch.utils.data import DataLoader
import streaming
import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# streaming.base.util.clean_stale_shared_memory()

tokenizer_path = '/fsx/yuli/cache/model/llama3_tokenizer'
# tokenizer_path = '/fsx/raymond/data/llama-2-7b-hf'
# tokenizer_path = 'lib/gemma-2b'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# local_path = '/fsx/yuli/cache/data/50B/llama3/en_hq'
# local_path = '/fsx/yuli/cache/data/50B/llama2/en_pile'
# local_path = '/fsx/yuli/cache/data/50B/llama2/id_hq'
# local_path = '/fsx/yuli/cache/data/50B/llama2/id_pile1'
# local_path = '/fsx/yuli/cache/data/50B/llama3/id_pile2'
# local_path = '/fsx/yuli/cache/data/50B/llama2/ms_hq'
local_path = '/fsx/yuli/cache/data/out_hero_llama3/th_pile1'
# local_path = '/fsx/yuli/cache/data/50B/llama2/exp1_p1'
local_path = '/fsx/yuli/cache/data/50B/llama3/id_hq'

# local_path = '/fsx/ngan/data/cpt/50B_high_quality/gemma_tokenizer/vi'
dataset = StreamingDataset(local=local_path, shuffle=False, keep_zip=True)

total_bs = len(dataset)
# idx = 0

start = 10000
end = 10005
for idx in range(total_bs):
    if idx >= start and idx < end:
        ids = np.frombuffer(dataset[int(f'{idx}')]['tokens'], dtype=np.int64).copy().tolist()
        # print(ids)
        tokens = tokenizer.decode(ids)
        subwords = tokenizer.convert_ids_to_tokens(ids)
        # print(tokens)
        # print(subwords)
        # print(ids)
        print(len(ids))
        print(len(tokens), len(ids), len(ids)/len(tokens))
        
        print('=========')
    if idx > end:
        break
