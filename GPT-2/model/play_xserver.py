from transformers import GPT2LMHeadModel
from transformers import pipeline, set_seed

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load the GPT-2 model
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

# Watch the layers specially tockens embeddings and position embeddings
for k, v in sd_hf.items():
    print(k, v.shape)

# Print some weights
print(sd_hf['transformer.wpe.weight'].view(-1)[:20])

# Plot the weights
plt.imshow(sd_hf['transformer.wpe.weight'], cmap='gray')
plt.show()

# Three random columns. What that channel is doing as a function of position
plt.plot(sd_hf['transformer.wpe.weight'][:, 150])
plt.plot(sd_hf['transformer.wpe.weight'][:, 200])
plt.plot(sd_hf['transformer.wpe.weight'][:, 250])
plt.show()

plt.imshow(sd_hf['transformer.h.1.attn.c_attn.weight'][:300,:300], cmap='gray')
plt.show()

generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)