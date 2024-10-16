import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import os
import opencc

data_path = 'data/'
df = pd.read_csv(os.path.join(data_path + '/old-newspaper.tsv'), sep='\t')
df[df['Language'] == 'Chinese (Traditional)'].shape
df = df[df['Language'] == 'Chinese (Traditional)'].iloc[:7000]
# 簡單做一下統計
df['len'] = df['Text'].apply(lambda x: len(str(x)))
print(df['len'].describe())
print(df[df['len'] <= 200].shape[0])
print(df[df['len'] >= 60].shape[0])
print(df[(df['len'] >= 60) & (df['len'] <= 200)].shape[0])
# 按照字串長度篩選一下資料集，可做可不做，看需求
df = df[(df['len'] >= 60) & (df['len'] <= 200)]
# 一個dict把中文字符轉化成id
char_to_id = {}
# 把id轉回中文字符
id_to_char = {}

# 有一些必須要用的special token先添加進來(一般用來做padding的token的id是0)
char_to_id['<pad>'] = 0
char_to_id['<eos>'] = 1
id_to_char[0] = '<pad>'
id_to_char[1] = '<eos>'

# 把所有資料集中出現的token都記錄到dict中
for char in set(df['Text'].str.cat()):
    ch_id = len(char_to_id)
    char_to_id[char] = ch_id
    id_to_char[ch_id] = char

vocab_size = len(char_to_id)
print('字典大小: {}'.format(vocab_size))
# 把資料集的所有資料都變成id
df['char_id_list'] = df['Text'].apply(lambda text: [char_to_id[char] for char in list(text)] + [char_to_id['<eos>']])
df[['Text', 'char_id_list']].head()
batch_size = 64
epochs = 3
embed_dim = 256
hidden_dim = 256
lr = 0.001
grad_clip = 1
# 這裏的dataset是文本生成的dataset，輸入和輸出的資料都是文章
# 舉個例子，現在的狀況是：
# input:  A B C D E F
# output: B C D E F <eos>
# 而對於加減法的任務：
# input:  1 + 2 + 3 = 6
# output: / / / / / 6 <eos>
# /的部分都不用算loss，主要是預測=的後面，這裏的答案是6，所以output是6 <eos>
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, index):
        # input:  A B C D E F
        # output: B C D E F <eos>
        x = self.sequences.iloc[index][:-1]
        y = self.sequences.iloc[index][1:]
        return x, y

    def __len__(self):
        return len(self.sequences)

def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch] # list[torch.tensor]
    batch_y = [torch.tensor(data[1]) for data in batch] # list[torch.tensor]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # torch.tensor
    # [[1968, 1891, 3580, ... , 0, 0, 0],
    #  [1014, 2242, 2247, ... , 0, 0, 0],
    #  [3032,  522, 1485, ... , 0, 0, 0]]
    #                       padding↑
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True, # shape=(batch_size, seq_len)
                                                  padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True, # shape=(batch_size, seq_len)
                                                  padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens
dataset = Dataset(df['char_id_list'])
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=collate_fn)

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        # Embedding層
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        # RNN層
        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        # output層
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        # 假設有個tensor : tensor([[1, 2, 3, 4],
        #                        [9, 0, 0, 0]])
        # 輸出就是：PackedSequence(data=tensor([1, 9, 2, 3, 4]),
        #                         batch_sizes=tensor([2, 1, 1, 1]),
        #                         sorted_indices=None, unsorted_indices=None)
        # torch.nn.utils.rnn.pack_padded_sequence 會把batch當中的句子從長到短排序，建立如上所示的資料結構
        # 就像上一個例子一樣，RNN會先吃第一個batch內的第一個batch_size，看到這個地方的batch_size爲2，所以此時RNN會吃兩個token，輸出一個2Xhidden_dim的向量組
        # 然後看第二個batch_size, 此時爲1，少了一個，說明其中一個序列到頭了，那就取上一個輸出向量的第一個，再生成一個1Xhidden_dim的向量
        # [
        # [1,2,3],                data = [1,4,6,2,5,3]   output  data = [1p,4p,6p,2p,5p,3p]             [1p,2p,3p]
        # [4,5,0], => pack_padded_sequence => batch_sizes = [3,2,1] => RNN => batch_sizes = [3,2,1] => pad_packed_sequence => [4p,5p,0]
        # [6,0,0]         _                                                     [6p,0,0]
        # ]
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):

        char_list = [char_to_id[start_char]]

        next_char = None

        # 生成的長度沒達到max_len就一直生
        while len(char_list) < max_len:
            x = torch.LongTensor(char_list).unsqueeze(0)
            x = self.embedding(x)
            _, (ht, _) = self.rnn_layer1(x)
            _, (ht, _) = self.rnn_layer2(ht)
            y = self.linear(ht)

            next_char = np.argmax(y.numpy())

            # 如果看到新的token是<eos>就說明生成結束了，就停下
            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]
    
    torch.manual_seed(2)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)
criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id['<pad>'], reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

from tqdm import tqdm
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    process_bar = tqdm(data_loader, desc=f"Training epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in process_bar:

        # 標準DL訓練幾板斧
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)
        batch_pred_y = batch_pred_y.view(-1, vocab_size)
        batch_y = batch_y.view(-1).to(device)
        loss = criterion(batch_pred_y, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

        i+=1
        if i%10==0:
            process_bar.set_postfix(loss=loss.item())

    # 麻煩各位同學加上 validation 的部分
    # validation_process_bar = tqdm(...)
    # for ... in validation_process_bar:
    #     pred = model...
    validation_process_bar = tqdm(data_loader, desc="Validation")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_lens, batch_y_lens in validation_process_bar:
            batch_pred_y = model(batch_x.to(device), batch_x_lens)
            batch_pred_y = batch_pred_y.view(-1, vocab_size)
            batch_y = batch_y.view(-1).to(device)
            loss = criterion(batch_pred_y, batch_y)
            total_loss += loss.item()
            validation_process_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {average_loss}")