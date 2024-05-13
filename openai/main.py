from openai import OpenAI
import numpy as np

import json

client = OpenAI()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# docs = [
#     {
#         "body": "Tokyo"
#     },
#     {
#         "body": "London"
#     },
#     {
#         "body": "Japan"
#     },
#     {
#         "body": "U.K."
#     }
# ]

docs = [
    # {
    #     'idx': 0,
    #     "body": "for (let i = 0; i < 10; i++) { if (i % 2 === 0) continue; console.log(i); }"
    # },
    # {
    #     'idx': 1,
    #     "body": "list.foreach((item) => { console.log(item); });"
    # },
    # {
    #     'idx': 2,
    #     "body": "err := dosomething(); if err != nil { return err; }"
    # },
    # {
    #     'idx': 3,
    #     "body": "final a = 1; final b = 2; print(a + b);"
    # },
    # {
    #     'idx': 4,
    #     "body": "aとbを定義して、aとbを足した結果を出力する"
    # },
    # {
    #     'idx': 5,
    #     "body": "0から9までの数字を出力するが、偶数の場合は出力しない"
    # },
    # {
    #     'idx': 6,
    #     "body": "const a = 1; const b = 2; console.log(a + b);"
    # },
    # {
    #     'idx': 7,
    #     'body': 'ミントグリーンのカーテンが軽やかに揺れ、日差しが心地よく室内を照らす。彼女は窓辺の椅子に座り、心地よい風を感じながら一息ついた'
    # },
    # {
    #     'idx': 8,
    #     'body': '緑色のカーテンが風に揺れ、部屋を優しく照らしている。彼は窓際に腰掛け、風を浴びながらひと休みした'
    # },
    # {
    #     'idx': 9,
    #     'body': 'カーテンの緑色が風になびき、光が部屋をやさしく照らし出す。彼は窓の傍で腰をかけ、風を享受しながら小休止した。'
    # },
    {
        'idx': 10,
        'body': '日本の首都は東京です。また、標準時からの時差はUTC+9です。',
    },
    {
        'idx': 11,
        'body': 'The capital of Japan is Tokyo. Also, the variation from standard time is UTC+9.',
    },
    {
        'idx': 12,
        'body': '日本では寿司が有名です。',
    },
    {
        'idx': 13,
        'body': 'Sushi is famous in Japan.',
    },
    {
        'idx': 14,
        'body': 'Attentionのインプットは、モデルの順伝播として流れてくる値とEncoderから出力されたmemoryと言われる値です。そして、Attentionの最終目的は入ってきたmemoryに対して適切なweightを掛け算して、推論すべき値（input）に対する「注意」を向けさせることです。'
    },
    {
        'idx': 15,
        'body': 'The inputs of attention are the values ​​flowing as forward propagation of the model and the values ​​output from the encoder called memory. The ultimate purpose of attention is to multiply the incoming memory by the correct weight and direct "attention" to the value (input) to be inferred.'
    },
    {
        'idx': 16,
        'body': 'Tokyo'
    },
    {
        'idx': 17,
        'body': 'Japan'
    },
    {
        'idx': 18,
        'body': 'paris',
    },
    {
        'idx': 19,
        'body': 'U.K.',
    },
]

embedded_vectors = []
for doc in docs:
    response = client.embeddings.create(
        model='text-embedding-3-large',
        input=doc['body'],
        parameters={
            'temperature': 0.1,
            'max_tokens': 512,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
    )

    embedded_vectors.append(np.array(response.data[0].embedding))

    print(len(response.data))
    print(len(response.data[0].embedding))

# # # 東京 - 日本 + イギリス vs ロンドン
# v1 = embedded_vectors[0] - embedded_vectors[2] + embedded_vectors[3]
# v2 = embedded_vectors[1]

# print(cos_sim(v1, v2))


# tokyo - japan + uk vs london
v1 = embedded_vectors[6] - embedded_vectors[7] + embedded_vectors[9]
v2 = embedded_vectors[8]

print(cos_sim(v1, v2))


# v1 = embedded_vectors[0]
# v2 = embedded_vectors[1]
# v3 = embedded_vectors[2]
# v4 = embedded_vectors[3]
# v5 = embedded_vectors[4]

# print(cos_sim(v1, v2))
# print(cos_sim(v1, v3))
# print(cos_sim(v1, v4))
# print(cos_sim(v1, v5))

for i, doc in enumerate(docs):
    for j, doc2 in enumerate(docs):
        if i >= j:
            continue
        r = cos_sim(embedded_vectors[i], embedded_vectors[j])
        if r > -0.6:
            print(f'{i} vs {j}: {r}')
