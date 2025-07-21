"""
MAP 대회 - 오픈소스 LLM용 추출 모듈
ModernBERT 및 Gemma 노트북에서 오픈소스 LLM(Gemma, Phi 등) 사용에 유용한 모듈들
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
import os
import keras
import keras_nlp

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


class GemmaModelManager:
    """Gemma 모델 관리 클래스 (Keras NLP 기반)"""

    def __init__(self, model_name: str = "gemma_2b_en", backend: str = "jax"):
        """
        Args:
            model_name: Gemma 모델명
            backend: Keras 백엔드 ('jax', 'torch', 'tensorflow')
        """
        self.model_name = model_name
        self.backend = backend
        self.gemma_lm = None
        self.optimizer = None

        # 환경 설정
        os.environ["KERAS_BACKEND"] = backend
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

        logger.info(f"Gemma 모델 매니저 초기화: {model_name} on {backend}")

    def load_model(self, sequence_length: int = 512):
        """
        Gemma 모델 로드

        Args:
            sequence_length: 시퀀스 길이 제한
        """
        try:
            self.gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(self.model_name)
            self.gemma_lm.preprocessor.sequence_length = sequence_length
            logger.info(f"Gemma 모델 로드 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"Gemma 모델 로드 실패: {e}")
            raise

    def enable_lora(self, rank: int = 64):
        """
        LoRA 활성화 (효율적 파인튜닝)

        Args:
            rank: LoRA 랭크
        """
        if self.gemma_lm is None:
            raise ValueError("먼저 모델을 로드해야 합니다.")

        self.gemma_lm.backbone.enable_lora(rank=rank)
        logger.info(f"LoRA 활성화 완료: rank={rank}")

    def setup_optimizer(self, learning_rate: float = 5e-5, weight_decay: float = 0.01):
        """
        옵티마이저 설정

        Args:
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
        """
        self.optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        # Layernorm과 bias는 weight decay에서 제외
        self.optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
        logger.info("옵티마이저 설정 완료")

    def compile_model(self):
        """모델 컴파일"""
        if self.gemma_lm is None or self.optimizer is None:
            raise ValueError("모델과 옵티마이저를 먼저 설정해야 합니다.")

        self.gemma_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self.optimizer,
            weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        logger.info("모델 컴파일 완료")

    def create_training_data(self, test_df: pd.DataFrame) -> List[str]:
        """
        York Yong의 템플릿을 사용한 훈련 데이터 생성

        Args:
            test_df: 테스트 데이터프레임

        Returns:
            훈련용 텍스트 리스트
        """
        MAP_dataset = []
        for index, row in test_df.iterrows():
            question, answer = row["QuestionText"], row["MC_Answer"]
            template = f"QuestionText:\n{question}\n\nMC_Answer:\n{answer}"
            MAP_dataset.append(template)

        logger.info(f"훈련 데이터 생성 완료: {len(MAP_dataset)}개 샘플")
        return MAP_dataset

    def train(self, training_data: List[str], epochs: int = 1, batch_size: int = 1):
        """
        모델 훈련

        Args:
            training_data: 훈련 데이터
            epochs: 에포크 수
            batch_size: 배치 크기
        """
        if self.gemma_lm is None:
            raise ValueError("먼저 모델을 로드하고 컴파일해야 합니다.")

        logger.info(f"훈련 시작: {epochs} 에포크, 배치 크기 {batch_size}")

        history = self.gemma_lm.fit(training_data, epochs=epochs, batch_size=batch_size)

        logger.info("훈련 완료")
        return history

    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_length: 최대 생성 길이

        Returns:
            생성된 텍스트
        """
        if self.gemma_lm is None:
            raise ValueError("먼저 모델을 로드해야 합니다.")

        response = self.gemma_lm.generate(prompt, max_length=max_length)
        return response


class GemmaPromptEngineer:
    """Gemma 전용 프롬프트 엔지니어링 클래스"""

    def __init__(self):
        self.templates = {
            "math_question": self._math_question_template,
            "comparison": self._comparison_template,
            "fraction": self._fraction_template,
            "step_by_step": self._step_by_step_template,
        }

    def _math_question_template(self, question: str, answer: str) -> str:
        """수학 문제 템플릿"""
        return f"QuestionText:\n{question}\n\nMC_Answer:\n{answer}"

    def _comparison_template(self, question: str, answer: str) -> str:
        """숫자 비교 템플릿"""
        return f"Which number is the greatest {answer} or {answer}?\n\nQuestion: {question}"

    def _fraction_template(self, question: str, answer: str) -> str:
        """분수 문제 템플릿"""
        return f"A triangle split into nine equal smaller triangles. Six of them are shaded. What fraction of the shape is not shaded?\n\nQuestion: {question}"

    def _step_by_step_template(self, question: str, answer: str) -> str:
        """단계별 추론 템플릿"""
        return f"Let's solve this step by step:\n\nQuestion: {question}\n\nAnswer: {answer}\n\nStep-by-step solution:"

    def create_prompt(
        self, question: str, answer: str, template_type: str = "math_question"
    ) -> str:
        """
        프롬프트 생성

        Args:
            question: 질문
            answer: 답변
            template_type: 템플릿 타입

        Returns:
            생성된 프롬프트
        """
        if template_type not in self.templates:
            raise ValueError(f"지원하지 않는 템플릿 타입: {template_type}")

        return self.templates[template_type](question, answer)


class GemmaDataAnalyzer:
    """Gemma 모델용 데이터 분석 클래스"""

    def __init__(self):
        self.train_data = None
        self.test_data = None

    def load_data(self, train_path: str, test_path: str):
        """
        데이터 로드

        Args:
            train_path: 훈련 데이터 경로
            test_path: 테스트 데이터 경로
        """
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

        logger.info(
            f"데이터 로드 완료: 훈련 {self.train_data.shape}, 테스트 {self.test_data.shape}"
        )

    def analyze_misconceptions(self):
        """오해 분석"""
        if self.train_data is None:
            raise ValueError("먼저 데이터를 로드해야 합니다.")

        # 오해 분포 분석
        misconception_counts = self.train_data["Misconception"].value_counts()

        # 상위 20개 오해
        top_20 = misconception_counts.head(20)

        # 하위 15개 오해
        bottom_15 = misconception_counts.tail(15)

        logger.info(f"오해 분석 완료: 총 {len(misconception_counts)}개 오해")
        logger.info(f"상위 20개 오해: {len(top_20)}개")
        logger.info(f"하위 15개 오해: {len(bottom_15)}개")

        return {
            "top_20": top_20,
            "bottom_15": bottom_15,
            "total_misconceptions": len(misconception_counts),
        }

    def analyze_categories(self):
        """카테고리 분석"""
        if self.train_data is None:
            raise ValueError("먼저 데이터를 로드해야 합니다.")

        category_counts = self.train_data["Category"].value_counts()
        logger.info(f"카테고리 분석 완료: {len(category_counts)}개 카테고리")

        return category_counts

    def get_sample_questions(self, n: int = 5):
        """
        샘플 질문 추출

        Args:
            n: 샘플 수

        Returns:
            샘플 질문들
        """
        if self.test_data is None:
            raise ValueError("먼저 데이터를 로드해야 합니다.")

        samples = self.test_data.head(n)
        return samples[["QuestionText", "MC_Answer", "StudentExplanation"]]
