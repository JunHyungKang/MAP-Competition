# Unslot을 이용한 DeepSeek 모델 파인튜닝 가이드

## 개요

이 문서는 Unslot 라이브러리를 사용하여 DeepSeek-R1-0528-Qwen3-8B 모델을 GRPO(Generative Reward-Powered Optimization) 방식으로 파인튜닝하는 방법을 설명합니다. 주요 목표는 수학 문제 해결 능력을 향상시키고, 인도네시아어로 추론 과정을 생성하도록 모델을 훈련하는 것입니다.

## 설치 및 설정

### 1. Unslot 설치

```python
# Colab 환경에서 설치
!pip install --no-deps unsloth vllm==0.8.5.post1

# 추가 의존성 설치 (Colab 전용)
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" huggingface_hub hf_transfer
```

### 2. 언어 감지 라이브러리 설치

```python
!pip install langid -qq
```

## 모델 로드 및 설정

### 1. FastLanguageModel을 사용한 모델 로드

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # 더 긴 추론 과정을 위해 증가 가능
lora_rank = 32  # 더 큰 랭크 = 더 똑똑하지만 느림

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-0528-Qwen3-8B",
    max_seq_length = max_seq_length,
    load_in_4bit = True,  # LoRA 16bit의 경우 False
    fast_inference = True,  # vLLM 빠른 추론 활성화
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,  # 메모리 부족시 감소
)
```

### 2. LoRA 설정

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,  # 0보다 큰 숫자 선택! 권장: 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2,  # *2는 훈련 속도 향상
    use_gradient_checkpointing = "unsloth",  # 메모리 사용량 감소
    random_state = 3407,
)
```

## 데이터 준비

### 1. 데이터셋 로드

```python
from datasets import load_dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
```

### 2. 시스템 프롬프트 설정

```python
system_prompt = """You are given a problem.
Think about the problem and provide your working out.
You must think in Bahasa Indonesia."""
```

### 3. 데이터 전처리

```python
# 데이터셋 매핑
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
})
```

### 4. 토큰 길이 필터링

```python
# 상위 90% 프롬프트 길이로 제한하여 잘림 방지
tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# 90% 최대 길이보다 작은 샘플만 필터링
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
```

## 보상 함수 정의

### 1. 형식 일치 보상 함수

```python
import re

# 정확한 형식 매칭 보상
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # 정확한 형식이 보이면 보상!
        if match_format.search(response) is not None: 
            score += 3.0
        scores.append(score)
    return scores

# 근사 형식 매칭 보상
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # 키워드 개수 세기 - 너무 많으면 페널티!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        scores.append(score)
    return scores
```

### 2. 답변 검증 보상 함수

```python
def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # 정답은 5점!
        if guess == true_answer:
            score += 5.0
        # 공백만 다른 경우는 적은 보상
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # 비율로 근사한 답변도 보상!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1: 
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: 
                    score += 1.5
                else: 
                    score -= 2.5  # 틀린 답변 페널티
            except:
                score -= 4.5  # 페널티
        scores.append(score)
    return scores
```

### 3. 언어 일관성 보상 함수

```python
import langid

def get_lang(text: str) -> str:
    if not text:
        return "und"
    lang, _ = langid.classify(text)
    return lang

def format_and_language_reward_func(completions, **kwargs):
    scores = []
    for completion_item in completions:
        if not completion_item or not isinstance(completion_item[0], dict) or "content" not in completion_item[0]:
            scores.append(-5.0)
            continue

        content = completion_item[0]["content"]
        lang = get_lang(content)

        if lang == 'id':  # 인도네시아어
            score = 5.0
        elif lang == 'en':  # 영어
            score = -3.0
        elif lang == 'zh':  # 중국어
            score = -3.0
        else:
            score = -5.0

        scores.append(score)
    return scores
```

## GRPO 훈련 설정

### 1. 훈련 파라미터 설정

```python
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,  # 더 부드러운 훈련을 위해 4로 증가
    num_generations = 4,  # 메모리 부족시 감소
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 100,  # 전체 훈련을 위해 1로 설정
    save_steps = 100,
    report_to = "none",  # Weights & Biases 사용 가능
    output_dir = "outputs",
)
```

### 2. 훈련 실행

```python
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
        format_and_language_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```

## 추론 및 테스트

### 1. LoRA 저장

```python
model.save_lora("grpo_lora")
```

### 2. 추론 테스트

```python
from vllm import SamplingParams

# 시스템 프롬프트 없이 테스트
messages = [
    {"role": "user", "content": "Solve (x + 2)^2 = 0"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 2048,
)

output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_lora"),
)[0].outputs[0].text
```

### 3. 시스템 프롬프트와 함께 테스트

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Solve (x + 2)^2 = 0"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_lora"),
)[0].outputs[0].text
```

## 모델 저장 옵션

### 1. float16으로 저장

```python
# 16bit로 병합
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# 4bit로 병합
model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit")
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# LoRA 어댑터만 저장
model.save_pretrained("model")
tokenizer.save_pretrained("model")
model.push_to_hub("hf/model", token = "")
tokenizer.push_to_hub("hf/model", token = "")
```

### 2. GGUF 변환

```python
# 8bit Q8_0로 저장
model.save_pretrained_gguf("model", tokenizer)
model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 16bit GGUF로 저장
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# q4_k_m GGUF로 저장
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 여러 GGUF 옵션으로 저장
model.push_to_hub_gguf(
    "hf/model",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
    token = "",
)
```

## 주요 특징

1. **빠른 훈련**: Unslot은 2배 빠른 무료 파인튜닝을 제공
2. **메모리 효율성**: 4bit 양자화와 그래디언트 체크포인팅으로 메모리 사용량 최적화
3. **다양한 보상 함수**: 형식 일치, 답변 정확도, 언어 일관성 등 다양한 보상 체계
4. **유연한 저장 옵션**: float16, 4bit, GGUF 등 다양한 형식으로 저장 가능
5. **vLLM 통합**: 빠른 추론을 위한 vLLM 지원

## 참고 자료

- [Unslot 공식 문서](https://docs.unsloth.ai/)
- [Unslot Discord](https://discord.gg/unsloth)
- [Unslot GitHub](https://github.com/unslothai/unsloth)
- [Open R1 Math 데이터셋](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) 