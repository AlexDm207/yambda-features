"""Ручной набор признаков: то, что выбрал бы инженер «по здравому смыслу».
Окна взяты «круглые» (1, 7, 30 дней и вся история) — именно этот выбор потом проверяет автопоиск.

Обозначения: u пользователь, i трек, a артист, al альбом, ui/ua/ual пары пользователя с ними.
Статистики: n прослушивания, full дослушанные (>=50%), rec показы рекомендаций, rec_full дослушанные
рекомендации, org органические, ratio сумма долей, sec секунды, like/dislike/unlike.
"""

USER = [
    "sum:u:n:1", "sum:u:n:7", "sum:u:n:30",
    "rate:u:full/n:7", "rate:u:full/n:30", "rate:u:full/n:inf",
    "rate:u:rec_full/rec:7", "rate:u:rec_full/rec:30", "rate:u:rec_full/rec:inf",  # «CTR» пользователя
    "rate:u:org/n:30",          # насколько сам выбирает музыку
    "rate:u:ratio/n:30",        # средняя доля прослушивания
    "rate:u:like/n:inf", "rate:u:dislike/n:inf",
    "sum:u:sec:7",
    "recency:u",
]

ITEM = [
    "static:length",
    "sum:i:n:7", "sum:i:n:30", "sum:i:n:inf",               # популярность
    "rate:i:full/n:7", "rate:i:full/n:30", "rate:i:full/n:inf",
    "rate:i:rec_full/rec:30", "rate:i:rec_full/rec:inf",    # «CTR» трека в рекомендациях
    "sum:i:rec:7",
    "rate:i:ratio/n:30",
    "rate:i:like/n:inf", "rate:i:dislike/n:inf",
    "rate:i:org/n:30",
    "age:i", "recency:i",
]

ARTIST_ALBUM = [
    "sum:a:n:30", "rate:a:full/n:30", "rate:a:full/n:inf",
    "rate:a:rec_full/rec:30", "rate:a:like/n:inf",
    "sum:al:n:30", "rate:al:full/n:30",
]

# прямые связи пользователь-трек
PAIR_DIRECT = [
    "sum:ui:n:7", "sum:ui:n:30", "sum:ui:n:inf",
    "sum:ui:full:inf", "rate:ui:full/n:inf",
    "sum:ui:rec:inf", "rate:ui:rec_full/rec:inf",
    "sum:ui:org:inf",
    "sum:ui:like:inf", "sum:ui:dislike:inf", "sum:ui:unlike:inf",
    "recency:ui", "age:ui",
]

# косвенные связи: пользователь-артист, пользователь-альбом
PAIR_INDIRECT = [
    "sum:ua:n:7", "sum:ua:n:30", "sum:ua:n:inf",
    "rate:ua:full/n:30", "rate:ua:full/n:inf",
    "rate:ua:rec_full/rec:inf",
    "share:ua:n:30", "share:ua:n:inf",
    "sum:ua:like:inf", "sum:ua:dislike:inf",
    "recency:ua",
    "sum:ual:n:inf", "rate:ual:full/n:inf", "recency:ual",
]

GROUPS = {"user": USER, "item": ITEM, "artist_album": ARTIST_ALBUM,
          "pair_direct": PAIR_DIRECT, "pair_indirect": PAIR_INDIRECT}
HANDCRAFTED = [f for g in GROUPS.values() for f in g]
assert len(HANDCRAFTED) == len(set(HANDCRAFTED))
