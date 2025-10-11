import os
import torch
import librosa
import numpy as np
import pickle
import sounddevice as sd
import scipy.io.wavfile as wav
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

# 語音樣本資料夾
DATA_PATH = r"C:\Users\h_0212\專題\voice_data"
DB_FILE = "speaker_db.pkl"

# 初始化 ECAPA-TDNN 
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# 函數：取 embedding
def get_embedding(file_path):
    signal, sr = librosa.load(file_path, sr=16000, mono=True)
    signal_tensor = torch.tensor(signal).unsqueeze(0)
    embedding = classifier.encode_batch(signal_tensor)
    return embedding.squeeze().detach().numpy()

# 建立資料庫並儲存
def build_database():
    speakers = ["bia", "hsu", "lin", "tsai", "xin"]
    speaker_embeddings = {}

    for spk in speakers:
        spk_folder = os.path.join(DATA_PATH, spk)
        emb_list = []
        for file in os.listdir(spk_folder):
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(spk_folder, file)
                emb = get_embedding(file_path)
                emb_list.append(emb)
        speaker_embeddings[spk] = np.mean(emb_list, axis=0)

    with open(DB_FILE, "wb") as f:
        pickle.dump(speaker_embeddings, f)

    print("資料庫建立並儲存完成")
    return speaker_embeddings

# 載入資料庫
def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            speaker_embeddings = pickle.load(f)
        print("已載入現有資料庫")
    else:
        speaker_embeddings = build_database()
    return speaker_embeddings

# 函數：辨識語音
def recognize_speaker(test_file, speaker_embeddings):
    test_emb = get_embedding(test_file).reshape(1, -1)
    scores = {}
    for spk, emb in speaker_embeddings.items():
        sim = cosine_similarity(test_emb, emb.reshape(1, -1))[0][0]
        scores[spk] = sim
    predicted = max(scores, key=scores.get)
    return predicted, scores

# 函數：從麥克風錄音並儲存
def record_from_mic(duration=3, filename="mic_test.wav"):
    fs = 16000  # 16kHz 取樣率（跟模型一致）
    print(f"請開始說話 ({duration} 秒)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    wav.write(filename, fs, (recording * 32767).astype(np.int16))
    print("錄音完成，檔案已存成", filename)
    return filename

#---------------
if __name__ == "__main__":
    # 載入資料庫
    speaker_embeddings = load_database()

    #操作部分
    while True:
        print("\n")
        print("1. 麥克風錄音辨識")
        print("2. 結束程式")
        choice = input("請選擇功能 (1-2): ")

        if choice == "1":
            mic_file = record_from_mic(duration=3)
            predicted, scores = recognize_speaker(mic_file, speaker_embeddings)

            print(f"\n辨識結果：{predicted}")
            print("辨識機率：")
            for spk, score in scores.items():
                print(f"  {spk}: {score:.4f}")
                
        elif choice == "2":
            print("程式結束")
            break

        else:
            print("輸入錯誤，重新輸入 (1-2)")
