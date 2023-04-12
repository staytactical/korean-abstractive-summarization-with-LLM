import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, PeftModel, TaskType, LoraConfig
import data_preprocessing
import model_training
import inference

def train():
    # 데이터 전처리 및 로더 생성
    TOKENIZER = AutoTokenizer.from_pretrained('lcw99/t5-large-korean-text-summary')
    DATA_PATH = './GPT_Competition/train_augmented.csv'
    TRAIN_SET = data_preprocessing.prepare_train_set(DATA_PATH, TOKENIZER)
    TRAIN_LOADER = data_preprocessing.create_train_loader(TRAIN_SET, batch_size=4)

    # 모델 생성
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained('lcw99/t5-large-korean-text-summary')
    PEFT_CONFIG = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    PEFT_MODEL = get_peft_model(BASE_MODEL, PEFT_CONFIG)
    PEFT_MODEL.to('cuda')

    # 모델 학습
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4
    model_training.train_model(PEFT_MODEL, TRAIN_LOADER, NUM_EPOCHS, LEARNING_RATE)


def inference():
    # 모델 불러오기
    PEFT_MODEL_PATH = "./GPT_Competition/T5_Lora_processed_{}epoch" # bset epoch
    TOKENIZER = AutoTokenizer.from_pretrained('lcw99/t5-large-korean-text-summary')
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained('lcw99/t5-large-korean-text-summary')
    MODEL = PeftModel.from_pretrained(model=BASE_MODEL, model_id=PEFT_MODEL_PATH)
    MODEL.to('cuda')

    # 추론 및 제출 파일 생성
    TEST_PATH = './GPT_Competition/test.csv'
    inference.create_submission_file(MODEL, TEST_PATH, TOKENIZER)

if __name__ == '__main__':
    
    mode = input("Choose mode (train/infer): ")
    
    if mode.lower() == "train":
        train()
    elif mode.lower() == "infer":
        inference()
    else:
        print("Invalid mode. Please enter 'train' or 'infer'.")
