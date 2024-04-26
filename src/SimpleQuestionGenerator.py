from typing import Any
import random


question_template_list = [
    """以下の文章をもとに、日本語で新しい問題を1つ作成してください。""",
    """以下の文章をもとに、日本語で新しい質問を1つ作成してください。""",
    """以下の文章をもとに、日本語で新しい指示を1つ作成してください。""",
]

class SimpleQuestionGenerator:
    def __init__(self,max_text_len=3000) -> None:
        self.max_text_len=max_text_len
        pass

    def __call__(self, text):
        inst=random.choice(question_template_list)

        #text=text[:random.randint(0,len(text))*2]
        #text=text[int(random.randint(0,len(text))/2):]
        text=text[:self.max_text_len]
        
        return inst+"\n"+text