# app/language_model.py
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")

def refine_sentence(char_sequence):
    input_text = "".join(char_sequence)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
