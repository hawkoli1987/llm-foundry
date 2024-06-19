from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

PAT = os.environ.get("HF_TOKEN")
print(PAT)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=PAT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", use_auth_token=PAT, trust_remote_code=True)

embedding_layer = model.model.embed_tokens
old_vocab_size, hidden_dim = embedding_layer.weight.shape

# get all the params
params = model.state_dict()
# get the original embeddings
embeddings = params["model.embed_tokens.weight"]
# average the embeddings
avg_embed = embeddings.mean(dim=0)

# get a new embedding layer weights
new_embeddings_weights = torch.stack(tuple((avg_embed for _ in range(128000))), dim=0)

# initialize a new embedding layer object with the weights
new_embedding_layer = torch.nn.Embedding(128000, hidden_dim, _weight=new_embeddings_weights)
new_embedding_layer.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)

# replace the existing embedding layer with the new embedding layer object
model.model.embed_tokens = new_embedding_layer


# average the lm_head
avg_lm_head = model.lm_head.weight.mean(dim=0)

# get a new lm_head layer weights
new_lm_head_weight = torch.stack(tuple((avg_lm_head for _ in range(128000))), dim=0)

# initialize a new lm_head layer object with the weights
new_lm_head = torch.nn.Linear(2048, 128000, bias=False)
new_lm_head.weight.data = new_lm_head_weight
model.lm_head = new_lm_head

# save the model
model.save_pretrained("gemma2b_128k_avg_embeddings")