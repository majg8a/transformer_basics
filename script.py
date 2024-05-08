with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

stoi  = { char: i for i,char in enumerate(chars) }
itos = { i: char for i,char in enumerate(chars) }

encode  = lambda s: [stoi[c] for c in s]
decode  = lambda l: ''.join([ itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))
