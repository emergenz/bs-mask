import torch
from tqdm import tqdm
from transformers import Qwen2Tokenizer
# from transformers import Qwen2ForCausalLM
from custom_transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
# from custom_transformers.models.qwen2.modeling_qwen2_no_causal import Qwen2ForCausalLM
from datasets import load_dataset

device = 'cuda'
model_id = 'Qwen/Qwen1.5-0.5B'
model = Qwen2ForCausalLM.from_pretrained(model_id).to(device)
tokenizer = Qwen2Tokenizer.from_pretrained(model_id)

test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = 1024  # Example chunk size (adjust as needed)
stride = 512      # Example stride size (adjust as needed)

lls = []
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(ppl.item())