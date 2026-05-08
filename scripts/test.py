from paddleformers.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
print(tokenizer.encode("中华人民共和国"))
