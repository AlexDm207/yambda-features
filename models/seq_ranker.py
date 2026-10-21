"""Трансформер над историей событий с запросом «пользователь + трек-кандидат».

  история (L последних событий до дня показа) -> Transformer encoder -> H
  кандидат: эмбеддинг трека + артиста + длина  -> q
  target attention: q смотрит на H (какие прошлые события похожи на этот трек)
  голова: MLP([q, attn, q*attn, среднее H]) -> логит дослушивания

Эмбеддинги треков общие для истории и кандидатов, поэтому «этот трек уже был в истории» модель
может увидеть через скалярное произведение в attention, без ручного признака.
История кодируется один раз на пользователя-день и переиспользуется для всех кандидатов дня.
"""
import torch
import torch.nn as nn

N_ECODE, N_ORG, N_AGO = 10, 3, 12


class SeqRanker(nn.Module):
    def __init__(self, n_items, n_art, d=128, L=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        self.art = nn.Embedding(n_art, d, padding_idx=0)
        self.ecode = nn.Embedding(N_ECODE, d, padding_idx=0)
        self.org = nn.Embedding(N_ORG, d, padding_idx=0)
        self.ago = nn.Embedding(N_AGO, d, padding_idx=0)
        self.pos = nn.Embedding(L + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))  # всегда видимый токен: история может быть пустой
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.len_proj = nn.Linear(1, d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 1))
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.art.weight, std=0.02)

    def forward(self, h_item, h_art, h_ecode, h_org, h_ago, c_item, c_art, c_len):
        """h_*: (B, L) — история, 0 = паддинг; c_*: (B, C) — кандидаты."""
        B, L = h_item.shape
        x = self.item(h_item) + self.art(h_art) + self.ecode(h_ecode) + self.org(h_org) + self.ago(h_ago)
        x = torch.cat([self.cls.expand(B, 1, -1), x], 1) + self.pos.weight[: L + 1]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), h_item == 0], 1)
        H = self.norm(self.enc(x, src_key_padding_mask=pad))
        valid = (~pad).unsqueeze(-1).float()
        pooled = (H * valid).sum(1) / valid.sum(1)
        q = self.item(c_item) + self.art(c_art) + self.len_proj(c_len.unsqueeze(-1))
        a, _ = self.cross(q, H, H, key_padding_mask=pad, need_weights=False)
        z = torch.cat([q, a, q * a, pooled.unsqueeze(1).expand_as(q)], -1)
        return self.head(z).squeeze(-1)
