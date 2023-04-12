import torch
from transformers import AutoTokenizer
import pandas as pd
import nltk
from tqdm.auto import tqdm
from datetime import datetime, timezone, timedelta

def predict(model, text_list, tokenizer):
    preds = []
    model.eval()
    for text in tqdm(text_list):
        with torch.no_grad():
            text.replace('\n', ' ')
            inputs = ['불필요한 내용은 생략하고 핵심 내용을 간단하게 한줄 요약: ' + text]
            inputs = tokenizer(inputs, max_length=1024, truncation=True, return_tensors="pt").to('cuda')
            output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            summary = nltk.sent_tokenize(decoded_output.strip())[0]
            preds.append(summary)
            print(summary, sep='\n----------\n',end='\n========\n') 
            
    return preds

def create_submission_file(model, test_path, tokenizer):
    test_df = pd.read_csv(test_path)
    text_list = test_df.text.tolist()
    preds = predict(model, text_list, tokenizer)
    test_df['summary'] = preds

    TIME_SERIAL = datetime.now(timezone(timedelta(hours=9))).strftime("%y%m%d-%H%M%S")
    SUBMISSION_PATH = f"./GPT_Competition/T5_Lora_aug_{TIME_SERIAL}.csv"
    test_df[['id', 'summary']].to_csv(SUBMISSION_PATH, index=False)
    print(SUBMISSION_PATH)
