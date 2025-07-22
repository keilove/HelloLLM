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
        print("emb:", emb)
    
        cos = emb.cos()
        sin = emb.sin()
        # 应用旋转
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
# 初始化参数
#第一部分测试
# batch = 2
# seq_len = 4
# dim = 8
# q = torch.randn(batch, seq_len, dim)
# k = torch.randn(batch, seq_len, dim)
# position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)  # [batch, seq_len]
# rope = RoPE(dim)
# q_rope, k_rope = rope.apply(q, k, position_ids)
# print("q_rope shape:", q_rope.shape)  # [2, 4, 8]
# print("k_rope shape:", k_rope.shape)




'''
第二部分
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 部分1: RoPE 旋转位置编码 (保持不变)
# ---------------------------------------------------------------------------
class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(2) # (bs, seq_len, 1, dim)
    sin = sin[position_ids].unsqueeze(2) # (bs, seq_len, 1, dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ---------------------------------------------------------------------------
# 部分2: 更精确的 MLA (Multi-head Latent Attention)
# ---------------------------------------------------------------------------
class DeepSeekMLA(nn.Module):
    """
    一个更忠实于论文描述的MLA实现，包含KV压缩、门控机制和输出整合。
    """
    def __init__(self, hidden_dim, num_heads, rope_dim, qk_rope_head_dim, kv_low_rank_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # QKV的头维度
        self.q_head_dim = hidden_dim // num_heads
        self.k_head_dim = hidden_dim // num_heads
        self.v_head_dim = hidden_dim // num_heads
        
        # RoPE作用的维度
        self.qk_rope_head_dim = qk_rope_head_dim
        
        # KV压缩后的低秩维度
        self.kv_low_rank_dim = kv_low_rank_dim

        # --- 核心投影层 ---
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.q_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.k_head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.v_head_dim, bias=False)
        
        # KV 压缩层
        self.k_compressor = nn.Linear(self.k_head_dim, self.kv_low_rank_dim, bias=False)
        self.v_compressor = nn.Linear(self.v_head_dim, self.kv_low_rank_dim, bias=False)
        
        # Q 分支投影 (用于注意力和门控)
        self.q_a_proj = nn.Linear(self.q_head_dim, self.kv_low_rank_dim, bias=False)
        self.q_gate_proj = nn.Linear(self.q_head_dim, self.v_head_dim, bias=False)

        # 注意力输出提升层
        self.o_v_proj = nn.Linear(self.kv_low_rank_dim, self.v_head_dim, bias=False)
        
        # 最终输出层 (Wo)
        self.o_proj = nn.Linear(num_heads * self.v_head_dim, hidden_dim, bias=False)

        self.rotary_emb = DeepSeekRotaryEmbedding(self.qk_rope_head_dim)

    def forward(self, hidden_states, position_ids):
        bs, seq_len, _ = hidden_states.shape

        # 1. 初始QKV投影
        q = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.q_head_dim)
        k = self.k_proj(hidden_states).view(bs, seq_len, self.num_heads, self.k_head_dim)
        v = self.v_proj(hidden_states).view(bs, seq_len, self.num_heads, self.v_head_dim)

        # 2. 对Q和K应用RoPE (在压缩前)
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q[..., :self.qk_rope_head_dim], k[..., :self.qk_rope_head_dim], cos, sin, position_ids)
        q = q.clone() # 避免原地修改
        k = k.clone()
        q[..., :self.qk_rope_head_dim] = q_rope
        k[..., :self.qk_rope_head_dim] = k_rope
        
        # 3. KV压缩
        k_compressed = self.k_compressor(k) # (bs, seq_len, num_heads, kv_low_rank_dim)
        v_compressed = self.v_compressor(v) # (bs, seq_len, num_heads, kv_low_rank_dim)

        # 4. Q分支处理
        # 4.1. Q投影到低秩空间以计算注意力
        q_a = self.q_a_proj(q) # (bs, seq_len, num_heads, kv_low_rank_dim)
        
        # 4.2. Q投影以形成门控
        q_gate = self.q_gate_proj(q) # (bs, seq_len, num_heads, v_head_dim)
        
        # 5. 低秩空间中的注意力计算
        # (bs, num_heads, seq_len, low_rank_dim) x (bs, num_heads, low_rank_dim, seq_len) -> (bs, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q_a.transpose(1, 2), k_compressed.transpose(1, 2).transpose(-2, -1)) / (self.kv_low_rank_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # (bs, num_heads, seq_len, seq_len) x (bs, num_heads, seq_len, low_rank_dim) -> (bs, num_heads, seq_len, low_rank_dim)
        attn_output_compressed = torch.matmul(attn_weights, v_compressed.transpose(1, 2))
        
        # 6. 注意力输出提升
        # (bs, seq_len, num_heads, v_head_dim)
        attn_output_uplifted = self.o_v_proj(attn_output_compressed).transpose(1, 2)
        
        # 7. 门控机制整合
        # 论文中的门控SwiGLU变体: silu(gate) * x
        gated_output = F.silu(q_gate) * attn_output_uplifted
        
        # 8. 最终输出投影 (Wo)
        gated_output = gated_output.reshape(bs, seq_len, self.num_heads * self.v_head_dim)
        final_output = self.o_proj(gated_output)
        
        return final_output

# ---------------------------------------------------------------------------
# 部分3: MoE (带共享专家) (保持不变)
# ---------------------------------------------------------------------------
class DeepSeekExpert(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))

class DeepSeekMoEWithSharedExpert(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_routed_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList([DeepSeekExpert(hidden_dim, ffn_dim) for _ in range(num_routed_experts)])
        self.shared_expert = DeepSeekExpert(hidden_dim, ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, dim)
        shared_expert_output = self.shared_expert(hidden_states_flat)
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = shared_expert_output
        for i in range(self.top_k):
            expert_indices = selected_experts[:, i]
            weights = routing_weights[:, i]
            for exp_idx in range(len(self.routed_experts)):
                mask = (expert_indices == exp_idx)
                if mask.any():
                    final_hidden_states[mask] += weights[mask].unsqueeze(-1) * self.routed_experts[exp_idx](hidden_states_flat[mask])
        return final_hidden_states.view(bs, seq_len, dim)

# ---------------------------------------------------------------------------
# 部分4: 完整的DeepSeek-V2解码器层 (整合新MLA)
# ---------------------------------------------------------------------------
class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, rope_dim, qk_rope_head_dim, kv_low_rank_dim, ffn_dim, num_experts, top_k):
        super().__init__()
        self.self_attn = DeepSeekMLA(hidden_dim, num_heads, rope_dim, qk_rope_head_dim, kv_low_rank_dim)
        self.mlp = DeepSeekMoEWithSharedExpert(hidden_dim, ffn_dim, num_experts, top_k)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)
        hidden_states = residual + ffn_output
        return hidden_states

# ===========================================================================
# 用例: 序列长度为 5
# ===========================================================================
print("--- 实例化DeepSeek-V2解码器层 (序列长度为5) ---")

# 定义超参数
hidden_dim = 128
num_heads = 4
qk_rope_head_dim = 32   # Q/K每个头的维度
kv_low_rank_dim = 16    # 压缩后的K/V维度
ffn_dim = 256           # 专家网络的中间层维度
num_routed_experts = 8  # 路由专家的数量
top_k = 2               # 每个token选择2个路由专家

# 实例化一个完整的解码器层
decoder_layer = DeepSeekV2DecoderLayer(
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    rope_dim=qk_rope_head_dim, # RoPE作用的维度
    qk_rope_head_dim=qk_rope_head_dim,
    kv_low_rank_dim=kv_low_rank_dim,
    ffn_dim=ffn_dim,
    num_experts=num_routed_experts,
    top_k=top_k
)
print("模型结构:\n", decoder_layer)

# 创建输入数据
batch_size = 1
seq_len = 5 # <--- 按您的要求，序列长度为5
input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

# 前向传播
output_tensor = decoder_layer(input_tensor, position_ids)

print("\n--- 前向传播结果 ---")
print(f"输入张量形状: {input_tensor.shape}")
print(f"位置ID形状:   {position_ids.shape}")
print(f"输出张量形状: {output_tensor.shape}")

# 验证输出形状是否正确
assert output_tensor.shape == input_tensor.shape
print("\n验证通过：输出形状与输入形状一致。")