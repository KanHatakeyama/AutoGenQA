import numpy as np

from llama_cpp import Llama

inst="""次のUserとAssistantのやりとりを0から9点の間で評価しなさい｡
基準: 正確に日本語で答えているかどうか"""


def prepare_prompt(q,a,instruct):
    question=f"""{instruct}
    #やりとり
    User:
    {q}
    Assistant:
    {a}
    #評価"""
    prompt = f"""<s>[INST]{question}[/INST] """
    return prompt

def parse_output(out):
    evaluations=out['choices'][0]["logprobs"]["top_logprobs"][0]
    eval_ints=[]
    for key in evaluations.keys():
        if key in ['0','1','2','3','4','5','6','7','8','9']:
            eval_ints.append(int(key))

    score=np.mean(eval_ints)

    return score

class GGUFEvaluator:
    def __init__(self,
        n_layers=300,
        n_ctx=4000,
        model_path="/home/hatakeyama/python/ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf",
    ):

        self.model= Llama(model_path = model_path,  n_ctx = n_ctx, n_gpu_layers=n_layers,logits_all=True )
    
    def __call__(self,q,a,instruct=inst):
        try:
            prompt=prepare_prompt(q,a,instruct)
            out=self.model.create_completion(prompt,max_tokens=1,logprobs=True)
            score=parse_output(out)
            return score
        except Exception as e:
            print("error",e)
            return -1
