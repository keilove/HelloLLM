'''
第一部分介绍传统的KV缓存，以序列长度4为例
第二部分介绍DeepSeek中的MLA技术
'''
#第一部分
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleKVcacheAttention(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, past_k=None, past_v=None):
        B, T, C = x.shape
        # 这是多头注意力对应的完整的qkv向量，维度 = 头数 * 头维度
        # 这里的qkv向量是没有分头的
        q = self.q_proj(x)      
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)     # [B, num_heads, T_total, head_dim]
            v = torch.cat([past_v, v], dim=2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T_total]
        attn_probs = F.softmax(attn_scores, dim=-1) 
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        return output, k, v
class SimpleDecoderLayer(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()
        self.attn = SimpleKVcacheAttention(embed_dim, num_heads)
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x, past_k=None, past_v=None):
        attn_out, k, v = self.attn(x, past_k, past_v)
        return self.ln(attn_out + x), k, v
class SimpleDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder = SimpleDecoderLayer(embed_dim, num_heads)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, input_ids, past_k=None, past_v=None):
        x = self.embed(input_ids)
        x, k, v = self.decoder(x, past_k, past_v)
        logits = self.lm_head(x)
        return logits, k, v
#模拟推理过程
def generate_sequence():
    vocab_size = 1000
    embed_dim = 128
    num_heads = 8
    max_len = 4

    model = SimpleDecoderModel(vocab_size, embed_dim, num_heads)

    #假设初始tokenBOS的tokenid是1
    generated = [1]
    past_k, past_v = None, None
    for step in range(max_len):
        input_ids = torch.tensor([[generated[-1]]])
        logits, past_k, past_v = model(input_ids, past_k, past_v)
        next_token = logits[:, :, :].argmax(dim=-1).item()
        generated.append(next_token)

        print(f"Step {step+1}: input={generated[-2]} → output={next_token}")
        print(f"  KV shape: k={past_k.shape}, v={past_v.shape}")

    print("Final generated sequence:", generated)




#第二部分 这部分实现DeepSeekV2中的技术，
#KV压缩缓存（把kv先降压并缓存），推理时候直接用fusion的矩阵避免重复解压
#Q压缩，通过独立线性层降低维度
#维度扩展RoPE,对q和k增加附加维度以编码位置信息
#矩阵融合优化，提前对计算好融合后的投影矩阵，以减轻推理计算量























# --------- 执行生成 ---------
if __name__ == "__main__":
    generate_sequence()




      

        