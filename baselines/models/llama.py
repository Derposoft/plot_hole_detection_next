"""
file containing the definitions for the baseline LLama model
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
BIG_MODEL = "georgesung/llama2_7b_chat_uncensored"
SMALL_MODEL = "EleutherAI/pythia-31m"

class Llama(nn.Module):
    def __init__(self, model_name="EleutherAI/pythia-31m", device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        self.prompt = "Some sentences in the following story have plot holes; return the numbers of those sentences as a comma-separated list: "

    def forward(self, x, kgs, documents: list[str], **kwargs):
        print(f"llm forward; input size: {len(documents)}")
        documents = [" ".join(doc) for doc in documents]
        documents = [self.prompt + doc for doc in documents]
        tokens = self.tokenizer(documents, return_tensors="pt").to(self.device) # , padding=True
        n_input_tokens = len(tokens.input_ids[0]) # (batch, seq_len, dim)
        print("llm gen")
        y = self.model.generate(tokens.input_ids, max_length=n_input_tokens+30)
        y = y[:, n_input_tokens:]
        y = self.tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("llm return value:", y)
        try:
            y = [int(z) for z in y.split(",")]
        except:
            print("hit bad path :(")
            y = [0]
        print("llm forward over; final y", y)
        return y
