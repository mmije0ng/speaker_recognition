import os
import numpy as np
import sounddevice as sd
from queue import Queue
from collections import deque
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine

# 설정
SAMPLE_RATE = 16000
FRAME_DURATION = 1
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
WINDOW_DURATION = 3
WINDOW_SIZE = FRAME_SIZE * WINDOW_DURATION
SIMILARITY_THRESHOLD = 0.60
EMBEDDING_PATH = "data/reference_multi.npy"

def verify_live_speaker(reference_embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    encoder = VoiceEncoder()
    q = Queue()
    buffer = deque(maxlen=WINDOW_SIZE)

    default_input = sd.default.device[0]

    def callback(indata, frames, time, status):
        if status:
            print(f"[!] 마이크 상태 경고: {status}")
        buffer.extend(indata[:, 0])
        if len(buffer) >= WINDOW_SIZE:
            q.put(np.array(buffer))

    with sd.InputStream(
        device=default_input,
        channels=1,
        dtype='float32',
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        callback=callback):
        
        print(f"\n🎙️ 실시간 화자 검증을 위해 {WINDOW_DURATION}초간 말해주세요... (Ctrl+C로 종료)")
        try:
            while True:
                if not q.empty():
                    audio_chunk = q.get()
                    processed = preprocess_wav(audio_chunk, SAMPLE_RATE)
                    test_embedding = encoder.embed_utterance(processed)
                    similarity = 1 - cosine(reference_embedding, test_embedding)
                    print(f"🧪 유사도: {similarity:.4f}")
                    return similarity > threshold
        except KeyboardInterrupt:
            print("⏹️ 중단됨")
            return False

def verify_speaker_by_name(speaker_name: str, embeddings_path: str = EMBEDDING_PATH) -> bool:
    if not os.path.exists(embeddings_path):
        print(f"[!] 임베딩 파일이 존재하지 않습니다: {embeddings_path}")
        return False

    embeddings = np.load(embeddings_path, allow_pickle=True).item()

    if speaker_name not in embeddings:
        print(f"[!] '{speaker_name}' 화자는 등록되어 있지 않습니다.")
        print("🗂️ 등록된 화자 목록:")
        for name in embeddings:
            print(f" - {name}")
        return False

    print(f"\n🎯 비교 대상 화자: {speaker_name}")
    reference_embedding = embeddings[speaker_name]
    return verify_live_speaker(reference_embedding)

if __name__ == "__main__":
    TARGET_SPEAKER = "박미정"  # 확인 대상 화자

    result = verify_speaker_by_name(TARGET_SPEAKER)

    if result:
        print(f"\n✅ 동일 화자로 검증됨: {TARGET_SPEAKER} (True)")
    else:
        print(f"\n❌ 동일 화자가 아님: {TARGET_SPEAKER} (False)")
