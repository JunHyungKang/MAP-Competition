# Gemma 노트북 분석: 수학 오해 극복을 위한 Gemma 모델 활용

## 📋 노트북 개요

**작성자**: Prata, Marília (mpwolke)  
**작성일**: 2025년 7월 10일  
**목적**: MAP 경쟁에서 Gemma 2B 모델을 활용한 수학 오해 예측

## 🎯 경쟁 과제 이해

### MAP (Misconception Annotation Project)
- **목표**: 학생들의 수학 오해를 정확히 예측하는 NLP 모델 개발
- **입력**: 학생들의 개방형 답변 설명
- **출력**: 잠재적 수학 오해 후보 제안
- **의의**: 교사들이 학생들의 잘못된 사고를 식별하고 해결하는 데 도움

### 오해 극복의 중요성
- 학생들은 자신의 지식이 잘못되었다는 것을 인식하지 못함
- 오해는 학생 사고에 깊이 뿌리박혀 있음
- 새로운 경험을 잘못된 이해를 통해 해석하여 학습 방해
- 일반적인 수업 방식(강의, 실험, 독서)만으로는 오해 극복 어려움

## 📊 데이터 분석

### 데이터셋 구조
- **훈련 데이터**: 36,696개 행, 7개 컬럼
- **테스트 데이터**: 3개 행
- **주요 컬럼**:
  - `QuestionText`: 수학 문제
  - `MC_Answer`: 학생이 선택한 객관식 답
  - `StudentExplanation`: 학생의 설명
  - `Category`: 오해 카테고리
  - `Misconception`: 식별된 수학 오해 (9,860개만 존재)

### 데이터 특징
- `Misconception` 컬럼에만 결측값 존재
- 다양한 오해 카테고리 분포
- 학생 설명의 다양성
- 상위 20개 오해와 하위 15개 오해의 분포 차이

## 🤖 Gemma 모델 활용

### 모델 설정
```python
# Keras NLP와 Gemma 2B 모델 로드
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

# LoRA 적용 (효율적 파인튜닝)
gemma_lm.backbone.enable_lora(rank=64)

# 시퀀스 길이 제한 (메모리 효율성)
gemma_lm.preprocessor.sequence_length = 512
```

### 훈련 설정
```python
# AdamW 옵티마이저
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Layernorm과 bias는 weight decay에서 제외
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# 모델 컴파일
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

### 데이터 전처리
```python
# York Yong의 템플릿 활용
MAP_dataset = []
for index, row in test.iterrows():
    question, answer = row['QuestionText'], row['MC_Answer']
    template = f"QuestionText:\n{question}\n\nMC_Answer:\n{answer}"
    MAP_dataset.append(template)
```

## 🚀 훈련 및 결과

### 훈련 과정
- **에포크**: 1
- **배치 크기**: 1
- **훈련 시간**: 4분 38초
- **손실**: 0.2589
- **정확도**: 55.59%

### 생성 결과 예시

#### 예시 1: 숫자 비교
**입력**: "Which number is the greatest 6 or 6.2?"  
**출력**: 
```
Step 1/2
First, we need to compare the whole numbers 6 and 6.2. Since 6 is a whole number, it is already the greatest number.

Step 2/2
Therefore, the answer is 6.
```

#### 예시 2: 분수 계산
**입력**: "A triangle split into nine equal smaller triangles. Six of them are shaded. What fraction of the shape is not shaded?"  
**출력**:
```
Step 1
1 of 2

The shaded triangles are 6 out of 9 triangles.

$\dfrac{6}{9}=\dfrac{2}{3} $

Result
2 of 2

$\dfrac{2}{3} $
```

## 💡 주요 인사이트

### 1. 모델의 특성
- **단계별 추론**: "Step 1/2", "Therefore" 등 구조화된 답변 생성
- **수학 표기법**: LaTeX 형식의 수학 표현식 사용
- **논리적 사고**: 단계별로 문제를 해결하는 과정 제시

### 2. 한계점
- **잘못된 추론**: 6.2 > 6임에도 6이 더 크다고 잘못 판단
- **제한된 컨텍스트**: 문제의 전체 맥락을 완전히 이해하지 못함
- **메모리 제약**: 512 토큰 제한으로 인한 입력 길이 제한

### 3. 개선 방향
- **더 정확한 프롬프트 엔지니어링**
- **더 많은 훈련 데이터**
- **더 큰 모델 사용**
- **더 정교한 후처리 로직**

## 🔧 기술적 구현

### 환경 설정
```python
# Keras 백엔드 설정
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# 필요한 라이브러리
import keras
import keras_nlp
```

### 메모리 최적화
- JAX 백엔드 사용
- LoRA를 통한 효율적 파인튜닝
- 시퀀스 길이 제한
- 배치 크기 최소화

## 📚 참고 자료

### 인용
- **경쟁**: MAP - Charting Student Math Misunderstandings
- **오해 극복 연구**: Joan Lucariello and David Naff
- **기술적 참고**: York Yong, Josh Longenecker, mpwolke

### 유용한 링크
- [경쟁 페이지](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/overview)
- [오해 극복 연구](https://www.apa.org/education-career/k12/misconceptions)
- [Gemma 문서](https://ai.google.dev/gemma/docs/lora_tuning#load_dataset)

## 🎯 결론

이 노트북은 Gemma 2B 모델을 활용하여 수학 오해 예측을 시도한 흥미로운 접근법을 보여줍니다. LoRA를 통한 효율적 파인튜닝과 구조화된 답변 생성 능력은 MAP 경쟁에서 유용할 수 있지만, 더 정확한 추론과 컨텍스트 이해를 위한 추가 개선이 필요합니다. 