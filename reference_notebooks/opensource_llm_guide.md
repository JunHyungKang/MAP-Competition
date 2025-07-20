# 오픈소스 LLM 사용 가이드 - MAP 대회

## 🎯 추출된 유용한 모듈들

ModernBERT 노트북에서 오픈소스 LLM(Gemma, Phi 등) 사용에 유용하게 활용할 수 있는 핵심 모듈들을 추출했습니다.

## 📦 주요 추출 모듈

### 1. **MAPDataProcessor** - 데이터 처리
```python
# 핵심 기능
- 데이터 로드 및 전처리
- 답의 정확성 특성 엔지니어링 (가장 중요!)
- 테스트 데이터에 정확성 특성 적용
```

### 2. **PromptEngineer** - 프롬프트 엔지니어링
```python
# 3가지 프롬프트 템플릿
- basic: 원본 노트북과 동일한 형식
- detailed: 오픈소스 LLM용 상세 프롬프트
- instruction: Gemma/Phi용 명령형 프롬프트
```

### 3. **TokenizationManager** - 토크나이징 관리
```python
# 주요 기능
- 토큰 길이 분석 및 시각화
- 배치 토크나이징
- 하드웨어 최적화 설정
```

### 4. **MAP3Metric** - 평가 메트릭
```python
# MAP@3 계산 및 제출 파일 생성
- 커스텀 MAP@3 메트릭
- 제출 형식 변환
```

### 5. **ModelTrainer** - 모델 훈련
```python
# 오픈소스 LLM 훈련 최적화
- 하드웨어 자동 감지
- fp16/bf16 자동 설정
```

## 🚀 오픈소스 LLM 추천 모델

### 1. **Microsoft Phi-2** (추천)
```python
model_name = "microsoft/phi-2"
# 장점: 작은 크기, 빠른 추론, 수학 능력 우수
# 크기: 2.7B 파라미터
```

### 2. **Google Gemma-2B**
```python
model_name = "google/gemma-2b"
# 장점: 안정적, 다국어 지원
# 크기: 2B 파라미터
```

### 3. **Microsoft Phi-1.5**
```python
model_name = "microsoft/phi-1.5"
# 장점: 빠른 훈련, 효율적
# 크기: 1.3B 파라미터
```

### 4. **TinyLlama**
```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# 장점: 매우 가벼움, 빠른 추론
# 크기: 1.1B 파라미터
```

## 💡 핵심 활용 전략

### 1. **특성 엔지니어링 우선순위**
```python
# 가장 중요한 특성: 답의 정확성
processor = MAPDataProcessor()
train_data = processor.engineer_correctness_feature(train_data)
```

### 2. **프롬프트 최적화**
```python
# 오픈소스 LLM용 상세 프롬프트
prompt_engineer = PromptEngineer()
prompts = prompt_engineer.create_prompts(train_data, template='detailed')
```

### 3. **토큰 길이 최적화**
```python
# 오픈소스 LLM은 더 긴 컨텍스트 지원
tokenizer_manager = TokenizationManager("microsoft/phi-2", max_length=512)
```

### 4. **하드웨어 최적화**
```python
# 자동 하드웨어 감지 및 설정
trainer = ModelTrainer("microsoft/phi-2", num_classes=65)
trainer.initialize_model()
```

## 🔧 사용 예시

### 기본 사용법
```python
from extracted_modules_for_opensource_llm import *

# 1. 데이터 처리
processor = MAPDataProcessor()
train_data = processor.load_and_preprocess_data("train.csv")
train_data = processor.engineer_correctness_feature(train_data)

# 2. 프롬프트 생성
prompt_engineer = PromptEngineer()
prompts = prompt_engineer.create_prompts(train_data, template='detailed')

# 3. 토크나이징
tokenizer_manager = TokenizationManager("microsoft/phi-2", max_length=512)
token_stats = tokenizer_manager.analyze_token_lengths(prompts)

# 4. 모델 훈련
trainer = ModelTrainer("microsoft/phi-2", num_classes=processor.n_classes)
trainer.initialize_model()
```

### 고급 사용법 (앙상블)
```python
# 여러 오픈소스 LLM 앙상블
models = [
    "microsoft/phi-2",
    "google/gemma-2b", 
    "microsoft/phi-1.5"
]

results = []
for model_name in models:
    trainer = ModelTrainer(model_name, num_classes=processor.n_classes)
    # 훈련 및 예측
    # results.append(predictions)
```

## ⚡ 성능 최적화 팁

### 1. **메모리 최적화**
```python
# 그래디언트 체크포인팅
training_args = TrainingArguments(
    gradient_checkpointing=True,
    per_device_train_batch_size=8,  # 작은 배치 크기
    gradient_accumulation_steps=4,   # 그래디언트 누적
)
```

### 2. **속도 최적화**
```python
# 혼합 정밀도 훈련
training_args = TrainingArguments(
    fp16=True,  # T4 GPU용
    bf16=False, # T4는 bf16 미지원
)
```

### 3. **정확도 최적화**
```python
# 학습률 스케줄링
training_args = TrainingArguments(
    learning_rate=3e-5,  # 더 작은 학습률
    warmup_steps=100,    # 워밍업
    weight_decay=0.01,   # 정규화
)
```

## 🎯 오픈소스 LLM의 장점

### 1. **인터넷 제한 없음**
- 캐글 코드 대회의 "인터넷 접근 비활성화" 제약 해결
- 완전 오프라인 환경에서 작동

### 2. **커스터마이징 자유도**
- 모델 아키텍처 수정 가능
- 특정 도메인에 맞는 파인튜닝

### 3. **비용 효율성**
- API 호출 비용 없음
- 무제한 사용 가능

### 4. **투명성**
- 모델 내부 동작 완전 이해
- 디버깅 및 개선 용이

## 🚨 주의사항

### 1. **모델 크기 제한**
- 캐글 환경의 메모리 제약 고려
- 작은 모델 우선 선택

### 2. **훈련 시간**
- 오픈소스 LLM은 훈련 시간이 길 수 있음
- 9시간 제한 내에서 완료 가능한 설정 필요

### 3. **하드웨어 호환성**
- Kaggle T4 GPU 특성 고려
- fp16 사용, bf16 비활성화

## 📊 예상 성능

### ModernBERT vs 오픈소스 LLM 비교
| 모델 | 크기 | 예상 CV | 장점 | 단점 |
|------|------|---------|------|------|
| ModernBERT-large | 355M | 0.938 | 검증된 성능 | API 의존 |
| Phi-2 | 2.7B | 0.92-0.95 | 완전 오프라인 | 메모리 사용량 |
| Gemma-2B | 2B | 0.90-0.93 | 안정적 | 느린 추론 |
| Phi-1.5 | 1.3B | 0.88-0.92 | 빠름 | 성능 제한 |

## 🎯 추천 접근법

1. **Phi-2로 시작** - 가장 균형잡힌 선택
2. **특성 엔지니어링 우선** - 답의 정확성 정보 활용
3. **프롬프트 최적화** - 상세한 프롬프트 사용
4. **앙상블 고려** - 여러 모델 조합

이 추출된 모듈들을 활용하면 오픈소스 LLM으로도 높은 성능을 달성할 수 있습니다! 