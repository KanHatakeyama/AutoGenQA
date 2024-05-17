# %%
# ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from datasets import load_dataset
import json
from datetime import datetime

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"data_multi_orca/completion_records{current_time_no_symbols}.jsonl"

print("init model")
bot = GGUFBot(model_path="../ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf",
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=50)
print("fin initiating model")


print("init original dataset")

# openorca
"""
ds = load_dataset(
    "atsushi3110/cross-lingual-openorcha-830k-en-ja", split="train")
df = pd.DataFrame(ds)
df["database"] = "atsushi3110/cross-lingual-openorcha-830k-en-ja_"+df["id/en"]
df["question"] = df["question/ja"]
df = df.drop(columns=["response/en", "system_prompt/en",
             "question/ja", "response/ja", "question/en", "id/en"], axis=1)
records = df.to_dict(orient='records')
"""

df = pd.read_csv("data/0514llmchat.csv")
df = df.drop(columns=["answer"], axis=1)
df = df[df["question"] == df["question"]]
df["database"] = "minnnade"
records = df.to_dict(orient='records')


while True:
    count = 0
    random.shuffle(records)
    for record in tqdm(records):
        q = record["question"]
        prompt = f"""以下のテンプレートに従って､日本語のマルチターンの指示データを生成してください
・応答1､指示2､応答2が空欄になっているので､埋めてください
・会話の中身は､ランダムに決定してください
・テンプレートは厳守すること
---
### 指示1:{q}
### 応答1:
### 指示2:
### 応答2:
"""
        try:
            a = bot.ask(prompt)
            if a == "":
                continue
            record["response"] = a
        except Exception as e:
            print(record, e)
            continue

        print(record)
        with open(out_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
