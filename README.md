# 🥉AI CONNECT 2023 노트북으로 GPT 맛보기(한국어 문서 생성 요약) 경진대회 3위
<img width="618" alt="3위" src="https://user-images.githubusercontent.com/76998855/233287208-6e06b73d-5b9f-46fd-96cf-850ddb63b1f6.PNG">


2023년 AI CONNECT에서 개최한 [노트북으로 GPT 맛보기 경진대회](https://aiconnect.kr/competition/detail/223/task/272/community/detail/113) 3위 학습 및 추론 코드.

## Model

[paust/pko-t5-large model](https://huggingface.co/paust/pko-t5-large) 모델을 AIHUB [요약문 및 레포트 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582)로 파인튜닝한 [lcw99/t5-large-korean-text-summary](https://huggingface.co/lcw99/t5-large-korean-text-summary)를 Base Model로 사용.

## Data

AIHUB [요약문 및 레포트 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582)에서 모든 종류(news_r, briefing, his_cul, paper, minute, edit, public, speech, literature, narration)의 Training 데이터 활용.

## Fine-tuning

거대 언어 모델 경량화 기법 중 하나인 [Low-Rank Adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf)를 Base Model에 적용하여 학습.

![lora-animated](https://user-images.githubusercontent.com/76998855/233287674-3d21d7f0-fa02-4cb6-b1f3-9c48daf931ff.gif)


### LoRA paramerter
```
r=8
lora_alpha=32
lora_dropout=0.1
```
### PEFT
허깅페이스의 [Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft) 라이브러리를 활용.

![image](https://user-images.githubusercontent.com/76998855/233291471-576871e1-4bb7-4503-a0d1-fdf916c7eb00.png)

PERT는 LoRA 외에도 T5 모델에 대해 prefix tuning도 지원하지만, prefix tuning보다 LoRA의 성능이 더 높았음. 

## Prompt

추론 시 원본 텍스트 앞에 "불필요한 내용은 생략하고 핵심 내용을 간단하게 한줄 요약:" Prompt를 사용.

**불필요한 내용은 생략하고 핵심 내용을 간단하게 한줄 요약:** {원본 텍스트}

