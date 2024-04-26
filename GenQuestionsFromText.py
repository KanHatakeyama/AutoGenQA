import argparse
import json
from src.GGUFBot import GGUFBot
from src.HFDataset import HFDataset
from src.SimpleQuestionGenerator import SimpleQuestionGenerator
from src.AnswerGenerator import AnswerGenerator

def main(args):
    ds = HFDataset(args.ds_name, streaming=True)
    for _ in range(args.preload_iter):
        next(ds)

    bot = GGUFBot(args.model_path, max_new_tokens=args.max_new_tokens, n_ctx=args.max_new_tokens, n_gpu_layers=args.n_layers)

    q_gen = SimpleQuestionGenerator()
    a_gen = AnswerGenerator(bot)

    save_path = f"data/{args.ds_name.replace('/', '_')}.jsonl"
    while True:
        try:
            record = {}
            record["text"] = next(ds)
            record["database"] = args.ds_name
            record["inst_question"] = q_gen(record["text"])
            record["question"] = bot.ask(record["inst_question"])
            a_gen(record)

            with open(save_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print("error", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with AI model.")
    parser.add_argument("--ds_name", type=str, default="hatakeyama-llm-team/WikiBookJa", help="Name of the dataset")
    parser.add_argument("--model_path", type=str, default="/home/hatakeyama/python/ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf", help="Path to the model")
    parser.add_argument("--n_layers", type=int, default=400, help="Number of model layers to be loaded on GPU")
    parser.add_argument("--max_new_tokens", type=int, default=4000, help="Maximum number of new tokens")
    parser.add_argument("--preload_iter", type=int, default=1000, help="Number of preload iterations")

    args = parser.parse_args()
    main(args)
