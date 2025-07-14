import torch
import math

class RoPE:
    def __init__(self, dim):
        self.dim = dim
        # 频率的计算（shape: [dim//2]）
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))  #(1,4)

    def rotate_half(self, x):
        """
        将输入的每一对偶数/奇数维度旋转
        x: [..., dim]
        return: [..., dim]
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply(self, q, k, position_ids):
        """
        q, k: [batch, seq_len, dim]
        position_ids: [batch, seq_len]
        return: q_embed, k_embed (with RoPE applied)
        """
        # position_ids: [batch, seq_len] → [batch, seq_len, 1]
        position_ids_expanded = position_ids[:, :, None].float()  # [B, L, 1]
        inv_freq_expanded = self.inv_freq[None, None, :]          # [1, 1, dim//2]

        # θ = pos * inv_freq → [B, L, dim//2] 在每个频率上的角度
        freqs = position_ids_expanded * inv_freq_expanded #(2,4,4)

        # 拼接成 full dim（即把每一对用来旋转的维度对接好）
        emb = torch.cat((freqs, freqs), dim=-1)  #(2,4,8) 
        cos = emb.cos()
        sin = emb.sin()
        # 应用旋转
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

# 初始化参数
batch = 2
seq_len = 4
dim = 8
q = torch.randn(batch, seq_len, dim)
k = torch.randn(batch, seq_len, dim)
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)  # [batch, seq_len]

rope = RoPE(dim)
q_rope, k_rope = rope.apply(q, k, position_ids)

print("q_rope shape:", q_rope.shape)  # [2, 4, 8]
print("k_rope shape:", k_rope.shape)