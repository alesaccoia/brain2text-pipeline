import time, torch, sys
sys.path.insert(0, '/mnt/20TB/asaccoia/brain2text/src')
from llm_score import score
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
m = 'facebook/opt-6.7b'
kw = dict(cache_dir='/mnt/20TB/asaccoia/brain2text/hf_cache')
tok = AutoTokenizer.from_pretrained(m, **kw); tok.padding_side = 'right'
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, **kw).eval()
sents = ['i think that we should go to the store tomorrow morning before it gets busy'] * 100
t = time.time(); s = score(model, tok, sents, 'cpu', 50); print('100 frasi CPU bf16:', time.time() - t, 's', s[:2])
