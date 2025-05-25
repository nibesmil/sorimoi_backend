import os
import tempfile
from flask import Flask, request, jsonify
from scorelogic import GPTScoringService

app = Flask(__name__)
scorer = GPTScoringService()

@app.route('/score', methods=['POST'])
def score():
    transcript = request.form.get('transcript')
    audio_file = request.files.get('audio')

    if not transcript or not audio_file:
        return jsonify({"error": "transcript and audio are required"}), 422

    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        result = scorer.evaluate(transcript, audio_path)
    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        result = {"score": 0, "feedback": "서버 내부 오류로 평가 실패"}

    # 임시 파일 삭제
    try:
        os.remove(audio_path)
    except Exception as e:
        print(f"⚠️ 임시 파일 삭제 실패: {e}")

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
