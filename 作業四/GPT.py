# 習題 4 -- 請用 AI 寫一個 GPT (Micro-GPT 實作範例)
# 班級：資工三  學號：111210544  姓名：葉俊成

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 超參數設定 (Hyperparameters)
batch_size = 32      # 同時處理的句子數量
block_size = 64      # GPT 的上下文最大長度 (Context length)
max_iters = 500      # 訓練迭代次數
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64          # 特徵維度 (Embedding dimension)
n_head = 4           # 多頭注意力機制的頭數 (Multi-head attention)
n_layer = 2          # Transformer Block 的層數

print(self_device := f"使用裝置: {device}")

# 2. 準備訓練文本 (這裡用一段簡單的文本作為範例資料)
text = "hello world! this is a micro gpt model built for machine learning class. artificial intelligence is powerful."
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 建立字元與數字互相轉換的字典 (Tokenizer)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# 切分訓練集
data = torch.tensor(encode(text), dtype=torch.long)

# 取得訓練批次資料的函式
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 3. GPT 核心組件實作
class Head(nn.Module):
    """ 單個注意力頭 (Single Head of Self-Attention) """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 計算注意力權重 (Scales attention scores)
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, T)
        # 因果遮罩 (Causal Masking)：確保模型看過前面字元，不能偷看未來的字
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) # (B, T, head_size)
        return wei @ v    # (B, T, head_size)

class MultiHeadAttention(nn.Module):
    """ 多頭注意力機制 (Multi-Head Attention) """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """ 簡單的線性前饋網路 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer 區塊：連接注意力與前饋網路 """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 4. MicroGPT 模型主體
class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # 混合文字與位置編碼
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ 給定初始上下文，生成接下來的文字 """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # 裁切到最大長度以防超出
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 5. 模型初始化與訓練
model = MicroGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- 開始訓練 MicroGPT 模型 ---")
for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"迭代步數 {iter:3d} | 當前 Loss 損失值: {loss.item():.4f}")

# 6. 測試文字生成效果
print("\n--- 模型訓練完成，開始進行文本生成測試 ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 以空字元當作起點
generated_output = decode(model.generate(context, max_new_tokens=50)[0].tolist())
print(f"生成的文字內容：\n{generated_output}")
