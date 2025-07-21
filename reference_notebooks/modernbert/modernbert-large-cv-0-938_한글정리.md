# ModernBERT Large - CV 0.938 노트북 분석

## 📋 노트북 개요

이 노트북은 MAP 대회에서 ModernBERT-large 모델을 사용하여 CV 0.938을 달성한 추론 전용 노트북입니다. 원본 DeBERTa 스타터 노트북의 후속작으로, 빠른 제출을 위한 추론만 수행하는 노트북입니다.

## 🎯 주요 접근 방법

### STEP 1: 모델 훈련
- `answerdotai/ModernBERT-large` 모델 사용
- 3 에포크로 CV 0.938 달성 (20% 홀드아웃 검증)
- 만족스러운 검증 점수 후 **100% 훈련 데이터로 재훈련**

### STEP 2: 추론 전용 노트북
- 훈련된 모델을 Kaggle 데이터셋에 업로드
- 이 노트북은 훈련 없이 추론만 수행
- 빠른 제출을 위한 최적화된 구조

## 🔧 기술적 세부사항

### 하드웨어 최적화
- **로컬 GPU**: `bf16=True`, `fp16=False` (새로운 GPU 지원)
- **Kaggle T4**: `bf16=False`, `fp16=True` (T4는 bf16 미지원)
- 하프 프리시전으로 훈련 및 추론 가속화

## 📊 데이터 처리

### 1. 데이터 로드 및 전처리
```python
# 라벨 인코딩
le = LabelEncoder()
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)  # 65개 클래스
```

### 2. 강력한 특성 엔지니어링
가장 중요한 특성: **답이 정확한지 여부**
```python
# 정답인 답변 찾기
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct['is_correct'] = 1

# 훈련 데이터에 병합
train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
```

## 📝 프롬프트 엔지니어링

### 입력 텍스트 포맷
```python
def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."  # 오타 있음
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )
```

### 프롬프트 구조
1. **질문 텍스트** 표시
2. **학생이 선택한 답** 표시
3. **답의 정확성** 표시 (정답/오답)
4. **학생 설명** 표시

## 🏗️ 모델 아키텍처

### 토크나이저 설정
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256
```

### 모델 초기화
```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    reference_compile=False,
)
```

## 🎯 훈련 설정

### 훈련 인수
```python
training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    num_train_epochs=EPOCHS,  # 3 에포크
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
    learning_rate=5e-5,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    fp16=True,  # Kaggle T4용
    bf16=False,  # T4는 bf16 미지원
)
```

## 📈 커스텀 MAP@3 메트릭

```python
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # 상위 3개 예측
    match = (top3 == labels[:, None])

    # MAP@3 수동 계산
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}
```

## 🔍 데이터 탐색 분석

### 질문 분석
- 15개의 객관식 수학 문제
- 각 문제당 4개 선택지 (A, B, C, D)
- 선택지는 인기도 순으로 정렬 (A가 가장 인기)

### 토큰 길이 분포
- 최대 토큰 길이: 256
- 256 토큰을 초과하는 샘플: 0개
- 모든 샘플이 토큰 제한 내에 포함

## 📤 제출 파일 생성

### 예측 처리
```python
# 상위 3개 예측 클래스 인덱스
top3 = np.argsort(-probs, axis=1)[:, :3]

# 숫자 클래스 인덱스를 원본 문자열 라벨로 디코딩
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# 3개 라벨을 공백으로 연결
joined_preds = [" ".join(row) for row in top3_labels]
```

### 제출 형식
```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36697,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
...
```

## 💡 핵심 인사이트

### 1. 특성 엔지니어링의 중요성
- **답의 정확성** 특성이 가장 중요한 특성
- 이 정보를 프롬프트에 명시적으로 포함

### 2. 프롬프트 설계
- 질문 → 답 → 정확성 → 학생 설명 순서
- 명확한 구조화된 입력이 성능 향상에 도움

### 3. 모델 선택
- ModernBERT-large가 우수한 성능
- 수학 도메인에 특화된 모델 활용

### 4. 평가 메트릭
- MAP@3에 최적화된 훈련
- 커스텀 메트릭으로 정확한 평가

## 🚀 개선 가능한 부분

1. **프롬프트 개선**: 오타 수정 및 더 나은 프롬프트 설계
2. **추가 특성**: 더 많은 특성 엔지니어링
3. **앙상블**: 여러 모델의 앙상블
4. **하이퍼파라미터 튜닝**: 더 정교한 튜닝

## 📚 참고 자료

- [원본 DeBERTa 스타터 노트북](https://www.kaggle.com/code/cdeotte/deberta-starter-cv-0-930/notebook)
- [ModernBERT 모델](https://huggingface.co/answerdotai/ModernBERT-large)
- [훈련된 모델 데이터셋](https://www.kaggle.com/datasets/cdeotte/modernbert-large-cv938) 