import pandas as pd
import json
import pandas as pd
import glob
from src.clean_records import clean_question
from huggingface_hub import HfApi, logging
import os
import time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def upload():
    jsonl_path_list=glob.glob('data/*.jsonl')

    all_records=[]
    for jsonl_path in jsonl_path_list:
        with open(jsonl_path) as f:
            for line in f:
                record=json.loads(line)
                all_records.append(record)

    original_record=all_records[0]
    cleaned_records=[]

    for original_record in all_records:
        record={}
        record["question"]=clean_question(original_record["question"])
        if "answer_1" not in original_record:
            original_record["answer_1"]=""

        if "ans0" in original_record:
            record["answer_0"]=original_record["ans0"]
            record["answer_1"]=original_record["ans1"]
        elif "answer_0" in original_record:
            record["answer_0"]=original_record["answer_0"]
            record["answer_1"]=original_record["answer_1"]
        else:
            print("no answer found",record)
        
        if "database" in original_record:
            record["database"]=original_record["database"]
        else:
            record["database"]="misc"

        
        cleaned_records.append(record)

    df=pd.DataFrame(cleaned_records)
    #シャッフル
    #df=df.sample(frac=1).reset_index(drop=True)
    parquet_path="hf/cleaned_data.parquet"
    df.to_parquet(parquet_path)
    df.to_csv("hf/cleaned_data.csv")



    logging.set_verbosity_debug()
    hf = HfApi()
    hf.upload_file(path_or_fileobj=parquet_path,
                    path_in_repo=f"1.parquet",
                    repo_id="hatakeyama-llm-team/AutoGeneratedJapaneseQA", repo_type="dataset")





if __name__ == "__main__":
    while True:
        try:
            upload()
            print("uploaded")
            time.sleep(3600*3)
        except Exception as e:
            print("error",e)
            time.sleep(600)

