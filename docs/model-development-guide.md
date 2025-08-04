# 🤖 모델 개발 가이드

이 가이드는 MAP-Competition 프로젝트에서 수학 오해 예측 모델을 개발하는 방법을 설명합니다.

## 📋 문서 개요

### 목적 및 범위
이 문서는 MAP-Competition 프로젝트에서 수학 오해를 예측하는 3단계 모델을 개발하는 방법을 안내합니다. 데이터 전처리부터 모델 훈련, 평가, 제출 파일 생성까지 모든 과정을 포함합니다.

### 주요 내용
- 3단계 예측 모델 아키텍처 설계
- 데이터 전처리 및 특성 엔지니어링
- 다양한 모델 접근법 (기본/고급/앙상블)
- 모델 훈련 및 성능 평가
- 제출 파일 생성 및 최적화

### 대상 독자
- 머신러닝 모델 개발에 관심이 있는 개발자
- 자연어 처리 및 텍스트 분류에 관심이 있는 연구자
- MAP 대회 참가자 및 데이터 사이언티스트

## 🚀 빠른 시작

### 핵심 단계 요약
1. **환경 설정 확인** (1분) - [환경 설정 가이드](README.md) 참조
2. **데이터 분석 완료** (시간 가변) - [EDA 가이드](../eda/README.md) 참조
3. **모델 아키텍처 설계** (30분-1시간)
4. **모델 훈련 및 평가** (1-3시간)
5. **제출 파일 생성** (30분)

### 주요 명령어
```bash
# 기본 모델 훈련
python train_basic_model.py

# 고급 모델 훈련
python train_advanced_model.py

# 앙상블 모델 훈련
python train_ensemble_model.py
```

### 예상 소요 시간
- **기본 모델 개발**: 2-3시간
- **고급 모델 개발**: 4-6시간
- **앙상블 모델 개발**: 6-8시간

## 🎯 모델 목표

## 🎯 모델 목표

### 3단계 예측 모델
1. **답변 정확성 판단**: True/False 예측
2. **오개념 포함 여부**: Correct/Misconception/Neither 예측
3. **구체적 오개념 식별**: 35개 오개념 유형 중 하나 예측

### 평가 지표
- **MAP@3**: Mean Average Precision @ 3
- 각 샘플당 최대 3개 예측 가능
- 예측 형식: `Category:Misconception`

## 📊 데이터 이해

### 입력 데이터
- **QuestionText**: 수학 문제 텍스트
- **MC_Answer**: 학생이 선택한 객관식 답
- **StudentExplanation**: 학생의 설명 텍스트

### 출력 데이터
- **Category**: 6개 클래스
  - True_Correct, True_Misconception, True_Neither
  - False_Correct, False_Misconception, False_Neither
- **Misconception**: 35개 오개념 유형 + NA

### 데이터 특징
- **총 샘플**: 36,696개 (훈련)
- **고유 문제**: 15개
- **텍스트 길이**: 평균 70자 (학생 설명)
- **불균형**: Category와 Misconception 분포 불균형

## 🏗️ 모델 아키텍처

### 1. 기본 접근법

#### 텍스트 분류 모델
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 초기화
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(category_labels)
)

# 입력 텍스트 구성
def prepare_input(question, answer, explanation):
    return f"Question: {question} Answer: {answer} Explanation: {explanation}"
```

#### 멀티태스크 학습
```python
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_categories, num_misconceptions):
        super().__init__()
        self.base_model = base_model
        self.category_classifier = nn.Linear(768, num_categories)
        self.misconception_classifier = nn.Linear(768, num_misconceptions)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        
        category_logits = self.category_classifier(pooled_output)
        misconception_logits = self.misconception_classifier(pooled_output)
        
        return category_logits, misconception_logits
```

### 2. 고급 접근법

#### 계층적 분류
```python
class HierarchicalClassifier:
    def __init__(self):
        # 1단계: True/False 분류
        self.true_false_classifier = self._build_classifier()
        
        # 2단계: Correct/Misconception/Neither 분류
        self.correct_misconception_classifier = self._build_classifier()
        
        # 3단계: 구체적 오개념 분류
        self.misconception_classifier = self._build_classifier()
    
    def predict(self, text):
        # 1단계 예측
        true_false = self.true_false_classifier.predict(text)
        
        # 2단계 예측
        correct_misconception = self.correct_misconception_classifier.predict(text)
        
        # 3단계 예측 (Misconception이 있는 경우만)
        if correct_misconception == "Misconception":
            misconception = self.misconception_classifier.predict(text)
        else:
            misconception = "NA"
        
        return f"{true_false}_{correct_misconception}:{misconception}"
```

#### 앙상블 모델
```python
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, text):
        predictions = []
        for model in self.models:
            pred = model.predict(text)
            predictions.append(pred)
        
        # 투표 또는 평균
        return self._ensemble_vote(predictions)
```

## 🔧 전처리 파이프라인

### 1. 텍스트 정제

#### 기본 정제
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    
    # 소문자 변환
    text = text.lower()
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)
```

#### 수학 표현식 처리
```python
def process_math_expressions(text):
    # LaTeX 표현식 보존
    latex_pattern = r'\\\(.*?\\\)'
    latex_matches = re.findall(latex_pattern, text)
    
    # 수학 표현식을 토큰으로 변환
    for i, match in enumerate(latex_matches):
        text = text.replace(match, f"MATH_EXPR_{i}")
    
    return text, latex_matches
```

### 2. 특성 엔지니어링

#### 텍스트 특성
```python
def extract_text_features(text):
    features = {}
    
    # 기본 통계
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()])
    
    # 수학 관련 특성
    features['math_symbols'] = len(re.findall(r'[+\-*/=<>]', text))
    features['numbers'] = len(re.findall(r'\d+', text))
    features['fractions'] = len(re.findall(r'\d+/\d+', text))
    
    return features
```

#### 감정 및 톤 분석
```python
from textblob import TextBlob

def extract_sentiment_features(text):
    blob = TextBlob(text)
    
    features = {}
    features['polarity'] = blob.sentiment.polarity
    features['subjectivity'] = blob.sentiment.subjectivity
    
    return features
```

### 3. 데이터 증강

#### 텍스트 증강
```python
from nlpaug.augmenter.word import SynonymAug
from nlpaug.augmenter.sentence import BackTranslationAug

def augment_text(text):
    # 동의어 치환
    synonym_aug = SynonymAug()
    augmented_texts = synonym_aug.augment(text, n=2)
    
    # 역번역
    back_translation_aug = BackTranslationAug()
    back_translated = back_translation_aug.augment(text, n=1)
    
    return augmented_texts + back_translated
```

## 🧠 모델 구현

### 1. BERT 기반 모델

#### 기본 BERT 분류기
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class MathMisconceptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=6):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    
    def train(self, train_dataloader, val_dataloader, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            self.model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
    
    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
        
        return predicted_class.item()
```

### 2. RoBERTa 기반 모델

#### RoBERTa 분류기
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class RoBERTaClassifier:
    def __init__(self, model_name='roberta-base', num_labels=6):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def prepare_input(self, question, answer, explanation):
        # RoBERTa 특화 입력 형식
        return f"Question: {question} <s> Answer: {answer} <s> Explanation: {explanation}"
```

### 3. DeBERTa 기반 모델

#### DeBERTa 분류기
```python
from transformers import DebertaTokenizer, DebertaForSequenceClassification

class DeBERTaClassifier:
    def __init__(self, model_name='microsoft/deberta-base', num_labels=6):
        self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
        self.model = DebertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
```

## 📈 학습 전략

### 1. 불균형 데이터 처리

#### 가중 손실 함수
```python
import torch.nn.functional as F

def weighted_loss(logits, labels, weights):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    weighted_ce_loss = ce_loss * weights[labels]
    return weighted_ce_loss.mean()

# 클래스별 가중치 계산
def calculate_class_weights(labels):
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights
```

#### 오버샘플링/언더샘플링
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def balance_dataset(X, y):
    # SMOTE 오버샘플링
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    return X_balanced, y_balanced
```

### 2. 교차 검증

#### Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(X, y, model_class, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_class()
        model.train(X_train, y_train)
        score = model.evaluate(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}: {score:.4f}")
    
    return np.mean(scores), np.std(scores)
```

### 3. 하이퍼파라미터 튜닝

#### Optuna를 사용한 최적화
```python
import optuna

def objective(trial):
    # 하이퍼파라미터 정의
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    max_length = trial.suggest_categorical('max_length', [256, 512, 768])
    
    # 모델 학습
    model = BERTClassifier()
    score = train_and_evaluate(model, lr, batch_size, max_length)
    
    return score

# 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 🎯 예측 및 후처리

### 1. Top-3 예측 생성

#### 확률 기반 예측
```python
def generate_top3_predictions(model, text):
    # 모델 예측
    logits = model.predict_proba(text)
    
    # Top-3 인덱스
    top3_indices = np.argsort(logits)[-3:][::-1]
    
    # 예측 형식 변환
    predictions = []
    for idx in top3_indices:
        category = category_labels[idx]
        misconception = misconception_labels[idx]
        predictions.append(f"{category}:{misconception}")
    
    return ' '.join(predictions)
```

#### 앙상블 예측
```python
def ensemble_predict(models, text):
    all_predictions = []
    
    for model in models:
        pred = model.predict_proba(text)
        all_predictions.append(pred)
    
    # 평균 확률
    avg_probs = np.mean(all_predictions, axis=0)
    
    # Top-3 선택
    top3_indices = np.argsort(avg_probs)[-3:][::-1]
    
    return format_predictions(top3_indices)
```

### 2. 후처리 규칙

#### 논리적 제약 조건
```python
def apply_logical_constraints(predictions):
    constrained_predictions = []
    
    for pred in predictions:
        category, misconception = pred.split(':')
        
        # 규칙 1: True_Correct에는 Misconception이 없어야 함
        if category == 'True_Correct' and misconception != 'NA':
            misconception = 'NA'
        
        # 규칙 2: False_Correct는 논리적으로 불가능
        if category == 'False_Correct':
            continue
        
        constrained_predictions.append(f"{category}:{misconception}")
    
    return constrained_predictions
```

## 📊 평가 및 분석

### 1. MAP@3 계산

#### 직접 구현
```python
def calculate_map_at_3(y_true, y_pred, k=3):
    """
    MAP@3 계산
    """
    def ap_at_k(y_true, y_pred, k):
        if len(y_pred) == 0:
            return 0.0
        
        # 정확한 예측이 있는지 확인
        correct = 0
        ap = 0.0
        
        for i, pred in enumerate(y_pred[:k]):
            if pred == y_true:
                correct += 1
                ap += correct / (i + 1)
        
        return ap / min(k, len(y_pred))
    
    aps = []
    for true, preds in zip(y_true, y_pred):
        ap = ap_at_k(true, preds, k)
        aps.append(ap)
    
    return np.mean(aps)
```

### 2. 오류 분석

#### 혼동 행렬 분석
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def analyze_errors(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return cm
```

#### 텍스트 패턴 분석
```python
def analyze_text_patterns(df, predictions, errors):
    # 오류가 많은 텍스트 패턴 분석
    error_texts = df[errors]['StudentExplanation']
    
    # 길이 분석
    error_lengths = error_texts.str.len()
    print(f"오류 텍스트 평균 길이: {error_lengths.mean():.2f}")
    
    # 자주 나오는 단어 분석
    error_words = ' '.join(error_texts).split()
    word_counts = Counter(error_words)
    print("오류 텍스트에서 자주 나오는 단어:")
    print(word_counts.most_common(10))
```

## 🚀 배포 및 제출

### 1. 모델 저장 및 로드

#### PyTorch 모델
```python
# 모델 저장
torch.save(model.state_dict(), 'best_model.pth')

# 모델 로드
model = BERTClassifier()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

#### ONNX 변환
```python
import torch.onnx

# ONNX 변환
dummy_input = torch.randn(1, 512)
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    export_params=True,
    opset_version=11
)
```

### 2. 제출 파일 생성

#### CSV 형식
```python
def create_submission_file(test_df, model):
    predictions = []
    
    for _, row in test_df.iterrows():
        text = prepare_input(
            row['QuestionText'],
            row['MC_Answer'],
            row['StudentExplanation']
        )
        
        pred = generate_top3_predictions(model, text)
        predictions.append(pred)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'Category:Misconception': predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    return submission_df
```

### 3. 성능 모니터링

#### 실시간 모니터링
```python
import wandb

def log_training_metrics(epoch, train_loss, val_loss, val_map):
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_map': val_map
    })
```

## 📚 다음 단계

모델 개발이 완료되면 다음 문서를 참조하여 고급 도구를 활용하세요:

1. **[도구 사용 가이드](README.md)**: LangChain과 LangGraph를 활용한 고급 모델 개발 및 워크플로우 구축

### 권장 진행 순서
1. **기본 모델 개발**: 이 가이드를 참조하여 기본 모델 구축
2. **성능 최적화**: 하이퍼파라미터 튜닝 및 앙상블 모델 구축
3. **고급 도구 활용**: [도구 사용 가이드](README.md)를 참조하여 LLM 워크플로우 구축
4. **최종 제출**: 모든 모델을 통합하여 최종 제출 파일 생성

## 📚 추가 리소스

### 유용한 링크
- [Transformers 공식 문서](https://huggingface.co/docs/transformers/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Scikit-learn 공식 문서](https://scikit-learn.org/)
- [Optuna 공식 문서](https://optuna.org/)

### 논문 및 참고 자료
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

## ✅ 체크리스트

모델 개발이 완료되었는지 확인하세요:

- [ ] 데이터 전처리 완료
- [ ] 모델 아키텍처 설계
- [ ] 학습 파이프라인 구축
- [ ] 하이퍼파라미터 튜닝
- [ ] 교차 검증 수행
- [ ] 앙상블 모델 구축
- [ ] 예측 후처리 구현
- [ ] 성능 평가 완료
- [ ] 제출 파일 생성
- [ ] 코드 정리 및 문서화

모든 항목이 체크되면 모델 개발이 완료된 것입니다! 🎉 