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

def identify_live_speaker(embeddings: dict, threshold: float = SIMILARITY_THRESHOLD):
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

        print(f"\n🎙️ 화자 식별을 위해 {WINDOW_DURATION}초간 말해주세요... (Ctrl+C로 종료)")
        try:
            while True:
                if not q.empty():
                    audio_chunk = q.get()
                    processed = preprocess_wav(audio_chunk, SAMPLE_RATE)
                    test_embedding = encoder.embed_utterance(processed)

                    best_match = None
                    best_similarity = -1

                    for name, ref_embedding in embeddings.items():
                        similarity = 1 - cosine(ref_embedding, test_embedding)
                        print(f"🔍 {name}: 유사도 {similarity:.4f}")
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name

                    if best_similarity > threshold:
                        print(f"\n✅ 예측된 화자: {best_match} (유사도 {best_similarity:.4f})")
                        return best_match
                    else:
                        print(f"\n❌ 등록된 화자와 일치하지 않음 (최대 유사도 {best_similarity:.4f})")
                        return None

        except KeyboardInterrupt:
            print("⏹️ 중단됨")
            return None

def identify_speaker(embeddings_path: str = EMBEDDING_PATH):
    if not os.path.exists(embeddings_path):
        print(f"[!] 임베딩 파일이 존재하지 않습니다: {embeddings_path}")
        return None

    embeddings = np.load(embeddings_path, allow_pickle=True).item()

    print(f"📦 등록된 화자 수: {len(embeddings)}명")
    return identify_live_speaker(embeddings)

if __name__ == "__main__":
    result = identify_speaker()

    if result:
        print(f"\n🎯 예측 결과: {result}")
    else:
        print("\n❌ 어떤 화자인지 식별하지 못했습니다.")
