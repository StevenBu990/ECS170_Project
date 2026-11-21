from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
import matplotlib
matplotlib.use('Agg')            # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import csv
import re
import time

# ---- experiment settings: change these for each run ----
RUN_NAME = 'lstm'
CELL = 'lstm'
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
EPOCHS = 100
LEARNING_RATE = 2e-3
BATCH_SIZE = 32
DEVICE = 'auto'            # 'auto' (GPU if available) or 'cpu'
N_COMPARE = 200            # how many dataset jokes to regenerate from their first 3 words
# three-word starts that are not necessarily in the dataset
RANDOM_STARTS = [['what', 'do', 'you'], ['why', 'did', 'the'], ['i', 'love', 'my'],
                 ['my', 'dog', 'is'], ['a', 'man', 'walks'], ['the', 'teacher', 'said']]
# ---------------------------------------------------------

RESULTS_DIR = '../../result/stage_4_result/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- parameter section ----
np.random.seed(2)
torch.manual_seed(2)

# ---- object initialization section ----
data_obj = Dataset_Loader_Jokes('jokes', '')
data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/data'
data = data_obj.load()
vocab = data_obj.vocab

method_obj = Method_RNN_Generator(f'{CELL.upper()} generator', '', vocab_size=len(vocab), device=DEVICE,
                                  cell=CELL, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
method_obj.max_epoch = EPOCHS
method_obj.learning_rate = LEARNING_RATE
method_obj.batch_size = BATCH_SIZE
method_obj.data = data

# ---- running section ----
print('************ Start ************')
start = time.time()
history = method_obj.run()
train_time = time.time() - start
print(f'training time: {train_time:.1f} s on {method_obj.device}')

SOS, EOS = vocab.word2idx['<sos>'], vocab.word2idx['<eos>']


def generate_tokens(words, greedy=True):
    ids = [vocab.word2idx.get(w, 1) for w in words]
    return vocab.decode(method_obj.generate(ids, SOS, EOS, greedy=greedy))


def to_text(tokens):
    return re.sub(r' ([?.!,])', r'\1', ' '.join(tokens))   # "cables ?" -> "cables?"


# ---- comparison with the training jokes (4-4) ----
rng = np.random.default_rng(2)
candidates = [j for j in data_obj.jokes if len(j) >= 4]
sample = [candidates[i] for i in rng.choice(len(candidates), size=min(N_COMPARE, len(candidates)), replace=False)]
rows, exact, word_acc = [], 0, []
for real in sample:
    gen = generate_tokens(real[:3])
    real_cont, gen_cont = real[3:], gen[3:]                  # compare only what the model generated
    exact += gen_cont == real_cont
    word_acc.append(sum(a == b for a, b in zip(gen_cont, real_cont)) / len(real_cont))
    rows.append({'type': 'dataset', 'start': ' '.join(real[:3]),
                 'generated': to_text(gen), 'real': to_text(real)})
exact_rate, mean_word_acc = exact / len(sample), float(np.mean(word_acc))

print('************ Generated vs. real (first 8 of the comparison set) ************')
for r in rows[:8]:
    print(f"start: {r['start']}\n  generated: {r['generated']}\n  real:      {r['real']}")
print(f'exact match: {exact_rate:.1%} of {len(sample)} jokes | word-by-word match after the start: {mean_word_acc:.1%}')

print('************ Starts not taken from the dataset ************')
for words in RANDOM_STARTS:
    unknown = [w for w in words if w not in vocab.word2idx]
    text = to_text(generate_tokens(words))
    print(f"{' '.join(words):18s} -> {text}" + (f'   (not in vocabulary: {unknown})' if unknown else ''))
    rows.append({'type': 'new start', 'start': ' '.join(words), 'generated': text, 'real': ''})

# ---- training plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(history['epoch'], history['loss'], color='#2a78d6')
ax1.set_title('Training loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-entropy loss')
ax1.grid(True, color='#e4e4e0')
ax2.plot(history['epoch'], history['acc'], color='#2a78d6')
ax2.set_title('Next-word accuracy (training)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True, color='#e4e4e0')
fig.suptitle(f'Joke generation training curves ({RUN_NAME})')
fig.tight_layout()
plt.savefig(f'{RESULTS_DIR}gen_{RUN_NAME}_curves.png', dpi=150)

# ---- record this run ----
pd.DataFrame(history).to_csv(f'{RESULTS_DIR}gen_{RUN_NAME}_history.csv', index=False)
pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}gen_{RUN_NAME}_examples.csv', index=False)
row = {'run': RUN_NAME, 'cell': CELL, 'embed_dim': EMBED_DIM, 'hidden_dim': HIDDEN_DIM,
       'num_layers': NUM_LAYERS, 'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE,
       'device': str(method_obj.device), 'time_s': round(train_time, 1),
       'final_loss': round(history['loss'][-1], 4), 'next_word_acc': round(history['acc'][-1], 4),
       'exact_match': round(exact_rate, 4), 'word_match': round(mean_word_acc, 4)}
table_path = RESULTS_DIR + 'generation_experiments.csv'
is_new = not os.path.exists(table_path)
with open(table_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if is_new:
        writer.writeheader()
    writer.writerow(row)
print('recorded run', RUN_NAME, 'in generation_experiments.csv')
print('************ Finish ************')
