"""
MAP 대회 - 오픈소스 LLM용 추출 모듈
ModernBERT 노트북에서 오픈소스 LLM(Gemma, Phi 등) 사용에 유용한 모듈들
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAPDataProcessor:
    """MAP 대회 데이터 처리 클래스"""
    
    def __init__(self):
        self.le = LabelEncoder()
        self.n_classes = None
        self.correct_answers = None
        
    def load_and_preprocess_data(self, train_path: str) -> pd.DataFrame:
        """
        훈련 데이터 로드 및 전처리
        
        Args:
            train_path: 훈련 데이터 파일 경로
            
        Returns:
            전처리된 훈련 데이터
        """
        # 데이터 로드
        train = pd.read_csv(train_path)
        train.Misconception = train.Misconception.fillna('NA')
        
        # 타겟 생성
        train['target'] = train.Category + ":" + train.Misconception
        train['label'] = self.le.fit_transform(train['target'])
        self.n_classes = len(self.le.classes_)
        
        logger.info(f"훈련 데이터 형태: {train.shape}, 클래스 수: {self.n_classes}")
        return train
    
    def engineer_correctness_feature(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        가장 중요한 특성: 답의 정확성 엔지니어링
        
        Args:
            train: 훈련 데이터
            
        Returns:
            정확성 특성이 추가된 데이터
        """
        # 정답인 답변 찾기 (Category가 'True'로 시작하는 것들)
        idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
        correct = train.loc[idx].copy()
        
        # 각 질문별로 가장 많이 선택된 답을 정답으로 간주
        correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
        correct = correct.sort_values('c', ascending=False)
        correct = correct.drop_duplicates(['QuestionId'])
        correct = correct[['QuestionId','MC_Answer']]
        correct['is_correct'] = 1
        
        # 훈련 데이터에 병합
        train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
        train.is_correct = train.is_correct.fillna(0)
        
        # 정답 정보 저장 (테스트 데이터용)
        self.correct_answers = correct
        
        logger.info("정확성 특성 엔지니어링 완료")
        return train
    
    def apply_correctness_to_test(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        테스트 데이터에 정확성 특성 적용
        
        Args:
            test: 테스트 데이터
            
        Returns:
            정확성 특성이 추가된 테스트 데이터
        """
        if self.correct_answers is None:
            raise ValueError("먼저 훈련 데이터로 정확성 특성을 엔지니어링해야 합니다.")
            
        test = test.merge(self.correct_answers, on=['QuestionId','MC_Answer'], how='left')
        test.is_correct = test.is_correct.fillna(0)
        return test


class PromptEngineer:
    """프롬프트 엔지니어링 클래스"""
    
    def __init__(self):
        self.prompt_templates = {
            'basic': self._basic_prompt,
            'detailed': self._detailed_prompt,
            'instruction': self._instruction_prompt
        }
    
    def _basic_prompt(self, row: pd.Series) -> str:
        """기본 프롬프트 (원본 노트북과 동일)"""
        x = "This answer is correct."
        if not row['is_correct']:
            x = "This answer is incorrect."  # 오타 수정
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"{x}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )
    
    def _detailed_prompt(self, row: pd.Series) -> str:
        """상세한 프롬프트 (오픈소스 LLM용)"""
        correctness = "correct" if row['is_correct'] else "incorrect"
        return (
            f"### Math Problem Analysis\n\n"
            f"**Question:** {row['QuestionText']}\n\n"
            f"**Student's Answer:** {row['MC_Answer']}\n"
            f"**Answer Status:** This answer is {correctness}.\n\n"
            f"**Student's Reasoning:** {row['StudentExplanation']}\n\n"
            f"### Task\n"
            f"Analyze the student's explanation and identify any mathematical misconceptions."
        )
    
    def _instruction_prompt(self, row: pd.Series) -> str:
        """명령형 프롬프트 (Gemma/Phi용)"""
        correctness = "correct" if row['is_correct'] else "incorrect"
        return (
            f"<|system|>\n"
            f"You are a math education expert. Analyze the student's explanation for mathematical misconceptions.\n"
            f"</s>\n"
            f"<|user|>\n"
            f"Question: {row['QuestionText']}\n"
            f"Student's Answer: {row['MC_Answer']} (This answer is {correctness})\n"
            f"Student's Explanation: {row['StudentExplanation']}\n"
            f"</s>\n"
            f"<|assistant|>\n"
        )
    
    def create_prompts(self, df: pd.DataFrame, template: str = 'basic') -> pd.Series:
        """
        데이터프레임에 프롬프트 적용
        
        Args:
            df: 데이터프레임
            template: 프롬프트 템플릿 ('basic', 'detailed', 'instruction')
            
        Returns:
            생성된 프롬프트 시리즈
        """
        if template not in self.prompt_templates:
            raise ValueError(f"지원하지 않는 템플릿: {template}")
            
        prompt_func = self.prompt_templates[template]
        return df.apply(prompt_func, axis=1)


class TokenizationManager:
    """토크나이저 관리 클래스"""
    
    def __init__(self, model_name: str, max_length: int = 256):
        """
        Args:
            model_name: 모델명 (예: 'microsoft/DialoGPT-medium', 'microsoft/phi-2')
            max_length: 최대 토큰 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 특수 토큰 추가 (필요한 경우)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def analyze_token_lengths(self, texts: pd.Series) -> Dict[str, Any]:
        """
        토큰 길이 분석
        
        Args:
            texts: 텍스트 시리즈
            
        Returns:
            토큰 길이 분석 결과
        """
        lengths = [len(self.tokenizer.encode(t, truncation=False)) for t in texts]
        
        # 히스토그램 그리기
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50)
        plt.title("Token Length Distribution")
        plt.xlabel("Number of tokens")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
        # 통계 정보
        stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'max': np.max(lengths),
            'min': np.min(lengths),
            'std': np.std(lengths),
            'exceed_limit': (np.array(lengths) > self.max_length).sum()
        }
        
        logger.info(f"토큰 길이 분석: {stats}")
        return stats
    
    def tokenize_batch(self, texts: pd.Series) -> Dict[str, torch.Tensor]:
        """
        배치 토크나이징
        
        Args:
            texts: 텍스트 시리즈
            
        Returns:
            토크나이징된 배치
        """
        return self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


class MAP3Metric:
    """MAP@3 메트릭 계산 클래스"""
    
    @staticmethod
    def compute_map3(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        MAP@3 메트릭 계산
        
        Args:
            eval_pred: (logits, labels) 튜플
            
        Returns:
            MAP@3 점수
        """
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # 상위 3개 예측
        top3 = np.argsort(-probs, axis=1)[:, :3]
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
    
    @staticmethod
    def create_submission(predictions: np.ndarray, test_df: pd.DataFrame, 
                         label_encoder: LabelEncoder) -> pd.DataFrame:
        """
        제출 파일 생성
        
        Args:
            predictions: 모델 예측 확률
            test_df: 테스트 데이터
            label_encoder: 라벨 인코더
            
        Returns:
            제출용 데이터프레임
        """
        # 상위 3개 예측 클래스 인덱스
        top3 = np.argsort(-predictions, axis=1)[:, :3]
        
        # 숫자 클래스 인덱스를 원본 문자열 라벨로 디코딩
        flat_top3 = top3.flatten()
        decoded_labels = label_encoder.inverse_transform(flat_top3)
        top3_labels = decoded_labels.reshape(top3.shape)
        
        # 3개 라벨을 공백으로 연결
        joined_preds = [" ".join(row) for row in top3_labels]
        
        # 제출 파일 생성
        submission = pd.DataFrame({
            "row_id": test_df.row_id.values,
            "Category:Misconception": joined_preds
        })
        
        return submission


class ModelTrainer:
    """모델 훈련 클래스"""
    
    def __init__(self, model_name: str, num_classes: int, device: str = 'auto'):
        """
        Args:
            model_name: 모델명
            num_classes: 클래스 수
            device: 디바이스 ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.trainer = None
        
    def initialize_model(self):
        """모델 초기화"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            reference_compile=False,
        )
        
        # 디바이스 설정
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        logger.info(f"모델 초기화 완료: {self.model_name} on {self.device}")
    
    def setup_training_args(self, output_dir: str, epochs: int = 3, 
                           batch_size: int = 16, learning_rate: float = 5e-5) -> TrainingArguments:
        """
        훈련 인수 설정
        
        Args:
            output_dir: 출력 디렉토리
            epochs: 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            
        Returns:
            훈련 인수
        """
        return TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            save_strategy="steps",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            logging_dir="./logs",
            logging_steps=50,
            save_steps=200,
            eval_steps=200,
            save_total_limit=1,
            metric_for_best_model="map@3",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to="none",
            # 하드웨어에 따른 설정
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        )


# 사용 예시
if __name__ == "__main__":
    # 1. 데이터 처리
    processor = MAPDataProcessor()
    train_data = processor.load_and_preprocess_data("train.csv")
    train_data = processor.engineer_correctness_feature(train_data)
    
    # 2. 프롬프트 엔지니어링
    prompt_engineer = PromptEngineer()
    prompts = prompt_engineer.create_prompts(train_data, template='detailed')
    
    # 3. 토크나이징 (오픈소스 LLM용)
    tokenizer_manager = TokenizationManager("microsoft/phi-2", max_length=512)
    token_stats = tokenizer_manager.analyze_token_lengths(prompts)
    
    # 4. 모델 훈련
    trainer = ModelTrainer("microsoft/phi-2", num_classes=processor.n_classes)
    trainer.initialize_model()
    
    print("모듈 추출 완료! 오픈소스 LLM 사용 준비됨.") 