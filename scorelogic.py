import os
import json
import librosa
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GPTScoringService:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def analyze_audio(self, filepath: str) -> dict:
        try:
            y, sr = librosa.load(filepath, sr=None)
            rms = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            silence_ratio = float(np.sum(rms < (0.1 * np.mean(rms))) / len(rms))
            try:
                pitches = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                avg_pitch = float(np.mean(pitches[~np.isnan(pitches)])) if pitches.size > 0 else 0.0
            except Exception:
                avg_pitch = 0.0
            return {
                "avg_rms": float(np.mean(rms)),
                "avg_zcr": float(np.mean(zcr)),
                "silence_ratio": silence_ratio,
                "avg_pitch": avg_pitch,
            }
        except Exception as e:
            raise RuntimeError(f"오디오 분석 실패: {e}")

    def generate_prompt(self, text: str, filename: str, metrics: dict) -> str:
        return f"""
다음은 사용자의 발음 평가를 위한 정보입니다.

🎙️ 인식된 문장:
{text}

🔊 오디오 분석 지표:
- 평균 음량 (RMS): {metrics['avg_rms']:.2f}
- 잡음 정도 (ZCR): {metrics['avg_zcr']:.4f}
- 무음 비율: {metrics['silence_ratio']:.2%}
- 평균 피치: {metrics['avg_pitch']:.2f} Hz

아래 항목을 고려하여 100점 만점으로 점수를 부여하고, JSON 형식으로 피드백을 작성해 주세요.
사용자마다 억양 및 톤이 다르니 일관된 피드백이 아닌 세세한 피드백으로 해주세요.

- 발음 정확도
- 말의 자연스러움
- 끊김이나 잡음 여부
- 전달력

응답 형식:
{{"score": 85, "feedback": "전반적으로 명확하지만 발음 속도가 빠릅니다."}}
"""

    def evaluate(self, text: str, filepath: str) -> dict:
        metrics = self.analyze_audio(filepath)
        prompt = self.generate_prompt(text, filepath, metrics)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            return {
                "score": int(result.get("score", 0)),
                "feedback": result.get("feedback", "")
            }
        except Exception as e:
            print(f"❌ GPT 응답 실패: {e}")
            return {
                "score": 0,
                "feedback": "GPT 응답 처리 중 오류가 발생했습니다."
            }
