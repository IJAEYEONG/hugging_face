from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import os
from huggingface_hub import login

# Hugging Face 로그인 (토큰 입력)
login("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")

# 모델 다운로드 경로 설정 (E 드라이브)
model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_directory = "E:/huggingface_models"

# 폴더가 없으면 생성
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 모델과 토크나이저 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory)

# 패딩 토큰 설정 (두 가지 방법 중 하나 선택)
tokenizer.pad_token = tokenizer.eos_token  # EOS 토큰을 패딩 토큰으로 사용
# 또는
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 새로운 PAD 토큰을 추가

# 학습할 JSON 데이터 준비
json_data = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "Who wrote 'Hamlet'?",
        "answer": "Hamlet was written by William Shakespeare."
    },
    {
        "question": "What is the boiling point of water?",
        "answer": "The boiling point of water is 100 degrees Celsius at sea level."
    }
]

# JSON 데이터를 텍스트 형식으로 변환 (QA 형식으로 변환)
data = [{"text": f"Q: {item['question']} A: {item['answer']}"} for item in json_data]

# 데이터셋 생성
dataset = Dataset.from_dict({"text": [item["text"] for item in data]})

# 토크나이저로 데이터 토크나이즈
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# 모델 학습
trainer.train()

# 학습 완료 후 모델 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"모델이 {save_directory}에 저장되었습니다.")
