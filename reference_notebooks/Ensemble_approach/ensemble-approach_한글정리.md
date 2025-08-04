# Ensemble Approach - 수학 오해 분석 모델 앙상블 방법론

## 개요
이 노트북은 MAP (Charting Student Math Misunderstandings) 대회에서 사용된 앙상블 접근법을 보여줍니다. 3개의 서로 다른 모델을 사용하여 학생들의 수학 오해를 분류하는 방법을 제시합니다.

## 데이터셋 정보
- **훈련 데이터**: 36,696개 샘플
- **타겟 클래스**: 65개 (Category:Misconception 조합)
- **문제 유형**: 15개의 다중 선택 수학 문제

## 1. Gemma2-9B-IT 모델

### 모델 설정
```python
model_name = "/kaggle/input/gemma2-9b-it-cv945"
EPOCHS = 2
```

### 특징
- **모델 크기**: 9B 파라미터
- **학습 방식**: LoRA (Low-Rank Adaptation) 어댑터 사용
- **데이터 타입**: bfloat16
- **배치 크기**: 훈련 8, 평가 16

### 프롬프트 형식
```
Question: {QuestionText}
Answer: {MC_Answer}
Correct? {Yes/No}
Student Explanation: {StudentExplanation}
```

### 성능
- CV 점수: 0.945

## 2. Ettin-Encoder-1B 모델

### 모델 설정
```python
model_name = "/kaggle/input/ettin-encoder-1b-cv943"
EPOCHS = 3
```

### 특징
- **모델 크기**: 1B 파라미터
- **모델 유형**: 인코더 전용 모델
- **배치 크기**: 훈련 32, 평가 64
- **학습률**: 5e-5

### 프롬프트 형식
```
Question: {QuestionText}
Answer: {MC_Answer}
Correct? {Yes/No}
Student Explanation: {StudentExplanation}
```

### 성능
- CV 점수: 0.943

## 3. ModernBERT-Large 모델

### 모델 설정
```python
model_name = "/kaggle/input/modernbert-large-cv938"
EPOCHS = 3
```

### 특징
- **모델 크기**: Large 버전
- **모델 유형**: BERT 기반
- **배치 크기**: 훈련 32, 평가 64
- **학습률**: 5e-5

### 프롬프트 형식
```
Question: {QuestionText}
Answer: {MC_Answer}
This answer is {correct/incorrect}.
Student Explanation: {StudentExplanation}
```

### 성능
- CV 점수: 0.938

## 핵심 특징 엔지니어링

### 정답 여부 특징
가장 중요한 특징 중 하나는 학생이 선택한 답이 정답인지 여부를 나타내는 `is_correct` 특징입니다.

```python
# 정답 답안 추출
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# 훈련 데이터에 정답 여부 병합
train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
```

## 평가 메트릭

### MAP@3 구현
대회의 평가 메트릭인 MAP@3를 직접 구현했습니다.

```python
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 예측
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

## 앙상블 방법론

### 가중 앙상블 알고리즘
3개 모델의 예측을 결합하는 가중 앙상블 방법을 사용했습니다.

```python
def get_top_k_ensemble(l1, l2, l3, k=3):
    list1, list2, list3 = l1.split('|'), l2.split('|'), l3.split('|')
    weights = [4, 4, 4]  # 신뢰도 가중치
    lists = [list1, list2, list3]
    score = defaultdict(int)

    for i, lst in enumerate(lists):
        weight = weights[i]
        for rank, item in enumerate(lst):
            score[item] += (len(lst) - rank) * weight

    # 점수 내림차순 정렬
    sorted_items = sorted(score.items(), key=lambda x: -x[1])
    return ' '.join([item for item, _ in sorted_items[:k]])
```

### 앙상블 과정
1. 각 모델별로 Top-3 예측 생성
2. 가중치를 적용하여 점수 계산
3. 최종 Top-3 예측 선택

## 메모리 관리

### GPU 메모리 정리
각 모델 훈련 후 메모리를 효율적으로 정리하는 코드가 포함되어 있습니다.

```python
import torch
import gc

# 모델 및 텐서 삭제
del model, trainer, predictions, probs

# GPU 캐시 정리
torch.cuda.empty_cache()
gc.collect()
torch.cuda.ipc_collect()
```

## 주요 인사이트

### 1. 모델 다양성
- **Gemma2**: 대용량 생성 모델, LoRA로 효율적 훈련
- **Ettin**: 경량 인코더 모델, 빠른 추론
- **ModernBERT**: BERT 기반, 안정적인 성능

### 2. 프롬프트 엔지니어링
- 정답 여부 정보가 중요한 특징
- 질문, 답안, 정답 여부, 학생 설명의 순서가 중요

### 3. 앙상블 효과
- 개별 모델보다 앙상블이 더 나은 성능
- 가중치 조정으로 모델별 기여도 조절 가능

### 4. 메모리 효율성
- LoRA 어댑터 사용으로 메모리 절약
- 모델 간 메모리 정리로 GPU 효율적 활용

## 결론

이 앙상블 접근법은 서로 다른 특성을 가진 3개 모델을 효과적으로 결합하여 수학 오해 분류 성능을 향상시켰습니다. 특히 정답 여부 특징과 MAP@3 메트릭의 직접 구현이 성능 향상에 크게 기여했습니다. 