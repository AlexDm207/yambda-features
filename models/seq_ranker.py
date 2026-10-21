"""Трансформер над историей событий с запросом «пользователь + трек-кандидат».

  история (L последних событий до дня показа) -> Transformer encoder -> H
  кандидат: эмбеддинг трека + артиста + длина  -> q
  target attention: q смотрит на H (какие прошлые события похожи на этот трек)
  голова: MLP([q, attn, q*attn, среднее H]) -> логит дослушивания

Эмбеддинги треков общие для истории и кандидатов, поэтому «этот трек уже был в истории» модель
может увидеть через скалярное произведение в attention, без ручного признака.
История кодируется один раз на пользователя-день и переиспользуется для всех кандидатов дня.

Ветка поиска по истории (retrieve > 0): для каждого кандидата модель дополнительно получает последние K
событий пользователя с этим же треком и с этим же артистом из ВСЕЙ истории (до дня показа) и число
таких событий. События кодируются теми же эмбеддингами, что и основная история, и сворачиваются
вниманием с запросом-кандидатом. Это замена ручных агрегатов «за всю историю» без ручных агрегатов.
"""
import torch
import torch.nn as nn

N_ECODE, N_ORG, N_AGO = 10, 3, 12


class SeqRanker(nn.Module):
    def __init__(self, n_items, n_art, d=128, L=256, layers=2, heads=4, dropout=0.1, retrieve=False):
        super().__init__()
        self.retrieve = retrieve
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
        n_in = 4 * d
        if retrieve:
            self.rq = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])     # запросы для трека и артиста
            self.cnt = nn.ModuleList([nn.Embedding(18, d) for _ in range(2)])  # log2(число совпадений)
            n_in += 2 * d
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(n_in, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 1))
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.art.weight, std=0.02)

    def events(self, item, art, ecode, org, ago):
        return self.item(item) + self.art(art) + self.ecode(ecode) + self.org(org) + self.ago(ago)

    def pool(self, j, tok, mask, cnt, q):
        """Внимание кандидата по найденным событиям: tok (B,C,K,d), mask (B,C,K), cnt (B,C)."""
        sc = torch.einsum("bckd,bcd->bck", tok, self.rq[j](q)) / tok.shape[-1] ** 0.5
        w = torch.softmax(sc.float().masked_fill(~mask, -1e4), -1) * mask
        return torch.einsum("bck,bckd->bcd", w.to(tok.dtype), tok) + self.cnt[j](cnt)

    def forward(self, h_item, h_art, h_ecode, h_org, h_ago, c_item, c_art, c_len, retr=None):
        """h_*: (B, L) — история, 0 = паддинг; c_*: (B, C) — кандидаты;
        retr: [(item, art, ecode, org, ago, mask, cnt) для совпадений по треку, то же по артисту]."""
        B, L = h_item.shape
        x = self.events(h_item, h_art, h_ecode, h_org, h_ago)
        x = torch.cat([self.cls.expand(B, 1, -1), x], 1) + self.pos.weight[: L + 1]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), h_item == 0], 1)
        H = self.norm(self.enc(x, src_key_padding_mask=pad))
        valid = (~pad).unsqueeze(-1).float()
        pooled = (H * valid).sum(1) / valid.sum(1)
        q = self.item(c_item) + self.art(c_art) + self.len_proj(c_len.unsqueeze(-1))
        a, _ = self.cross(q, H, H, key_padding_mask=pad, need_weights=False)
        z = [q, a, q * a, pooled.unsqueeze(1).expand_as(q)]
        if self.retrieve:
            for j, (it, ar, ec, og, ag, m, cnt) in enumerate(retr):
                z.append(self.pool(j, self.drop(self.events(it, ar, ec, og, ag)), m, cnt, q))
        return self.head(torch.cat(z, -1)).squeeze(-1)
