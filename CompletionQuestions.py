# %%
#ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from src.AnswerGenerator import AnswerGenerator
import json
import os
import copy
bot = GGUFBot(model_path="/home/hatakeyama/python/ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf", 
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=400)
#bot = GGUFBot(args.model_path, max_new_tokens=args.max_new_tokens, n_ctx=args.max_new_tokens, n_gpu_layers=args.n_layers)



out_path="data/completion_records.jsonl"
# %%

a_gen = AnswerGenerator(bot,n_answers=1)

def load_questions():
    df=pd.DataFrame()
    temp_df=pd.read_csv("data/raw_qa/0426spread.csv")
    df=pd.concat([df,temp_df])


    path_list=[
        "data/raw_qa/alpaca.jsonl",
        "data/raw_qa/t1.jsonl",
        "data/raw_qa/t2.jsonl",
        "data/raw_qa/a100/out.jsonl",
        "data/public/databricks-dolly-15k-ja_conv.jsonl",
        "data/public/oasst_ja_conv.jsonl",
    ]
    for path in path_list:
        temp_df= pd.read_json(path, orient='records', lines=True)
        temp_df["database"]=path
        df=pd.concat([df,temp_df])

    records=df.to_dict(orient='records')
    random.shuffle(records)


    if os.path.exists(out_path):
        done_df=pd.read_json(out_path, orient='records', lines=True)
        done_questions=list(done_df["question"].values)
    else:
        done_questions=[]

    undone_questions=[]
    for record in records:
        if record["question"] not in done_questions:
            undone_questions.append(record)
    return undone_questions,done_questions
    return records,done_questions

count=0
while True:
    print("start loop")
    records,done_questions=load_questions()
    for record in tqdm(records):
        if "q" in  record.keys():
            if type(record["q"]) is not str:
                continue
            record["question"]=record["q"]
            record["answer"]=record["a"]
            #print(record)
        if record["question"] in done_questions:
            print(f"skip {record['question']}")
            continue
        a_gen(record)
        record["answer_1"]=copy.copy(record["answer"])
        record.pop("answer")
        with open(out_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')

        count+=1
        if count>100:
            break