import os
import librosa
import soundfile as sf
import numpy as np
import random
from pathlib import Path
import argparse

class VoiceDataAugmenter:
    def __init__(self, input_dir, output_dir, sample_rate=16000):
        """
        語音資料擴增器初始化
        
        Args:
            input_dir: 原始MP3檔案資料夾路徑
            output_dir: 輸出資料夾路徑
            sample_rate: 取樣率（預設16000Hz）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # 創建輸出資料夾
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_audio(self, file_path):
        """載入音訊檔案"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"載入檔案失敗 {file_path}: {e}")
            return None, None
    
    def random_crop(self, audio, crop_length_seconds=3.0):
        """
        隨機裁剪音訊
        
        Args:
            audio: 音訊資料
            crop_length_seconds: 裁剪長度（秒）
        """
        crop_length = int(crop_length_seconds * self.sample_rate)
        
        if len(audio) <= crop_length:
            # 如果音訊太短，則重複填充
            repeat_times = (crop_length // len(audio)) + 1
            audio = np.tile(audio, repeat_times)
        
        # 隨機選擇起始點
        start_idx = random.randint(0, len(audio) - crop_length)
        return audio[start_idx:start_idx + crop_length]
    
    def add_white_noise(self, audio, noise_factor=0.005):
        """
        添加白雜訊
        
        Args:
            audio: 音訊資料
            noise_factor: 雜訊強度（0.001-0.01較為適中）
        """
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise
    
    def add_pink_noise(self, audio, noise_factor=0.005):
        """
        添加粉紅雜訊（1/f雜訊）
        
        Args:
            audio: 音訊資料
            noise_factor: 雜訊強度
        """
        # 生成粉紅雜訊
        uneven = len(audio) % 2
        X = np.random.randn(len(audio) // 2 + 1 + uneven) + 1j * np.random.randn(len(audio) // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        
        # 正規化並調整強度
        y = y / np.std(y) * noise_factor
        return audio + y[:len(audio)]
    
    def time_stretch(self, audio, stretch_factor=None):
        """
        時間拉伸（改變語速但不改變音調）
        
        Args:
            audio: 音訊資料
            stretch_factor: 拉伸係數（0.8-1.2較為適中）
        """
        if stretch_factor is None:
            stretch_factor = random.uniform(0.8, 1.2)
        
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    def pitch_shift(self, audio, n_steps=None):
        """
        音調轉換
        
        Args:
            audio: 音訊資料
            n_steps: 半音階數（-2到+2較為適中）
        """
        if n_steps is None:
            n_steps = random.uniform(-2, 2)
        
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def process_single_file(self, input_file, speaker_id, augment_count=2):
        """
        處理單個檔案並生成擴增資料
        """
        print(f"處理檔案: {input_file}")
        
        # 載入音訊
        audio, sr = self.load_audio(input_file)
        if audio is None:
            return
        
        # 原始檔名（不含副檔名）
        base_name = input_file.stem  

        # 創建說話者資料夾
        speaker_dir = self.output_dir / f"speaker_{speaker_id}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始檔案（命名為 xxx.0.wav）
        original_output = speaker_dir / f"{base_name}.0.wav"
        sf.write(original_output, audio, self.sample_rate)
        print(f"保存原始檔案: {original_output}")
        
        # 生成擴增資料
        for i in range(augment_count):
            try:
                # 隨機選擇擴增方法
                augmentation_methods = []
                
                if random.random() > 0.3:
                    crop_length = random.uniform(2.0, 5.0)
                    augmentation_methods.append(('crop', crop_length))
                
                noise_type = random.choice(['white', 'pink'])
                noise_factor = random.uniform(0.002, 0.008)
                augmentation_methods.append(('noise', noise_type, noise_factor))
                
                if random.random() > 0.7:
                    stretch_factor = random.uniform(0.9, 1.1)
                    augmentation_methods.append(('stretch', stretch_factor))
                
                if random.random() > 0.8:
                    n_steps = random.uniform(-1, 1)
                    augmentation_methods.append(('pitch', n_steps))
                
                # 應用擴增方法
                augmented_audio = audio.copy()
                for method in augmentation_methods:
                    if method[0] == 'crop':
                        augmented_audio = self.random_crop(augmented_audio, method[1])
                    elif method[0] == 'noise':
                        if method[1] == 'white':
                            augmented_audio = self.add_white_noise(augmented_audio, method[2])
                        else:
                            augmented_audio = self.add_pink_noise(augmented_audio, method[2])
                    elif method[0] == 'stretch':
                        augmented_audio = self.time_stretch(augmented_audio, method[1])
                    elif method[0] == 'pitch':
                        augmented_audio = self.pitch_shift(augmented_audio, method[1])
                
                # 正規化音量
                augmented_audio = augmented_audio / np.max(np.abs(augmented_audio)) * 0.9
                
                # 檔名：原始名稱 + 流水號
                output_filename = f"{base_name}.{i+1}.wav"
                output_path = speaker_dir / output_filename
                
                # 保存擴增後的音訊
                sf.write(output_path, augmented_audio, self.sample_rate)
                print(f"生成擴增檔案: {output_path}")
                
            except Exception as e:
                print(f"處理第 {i+1} 個擴增資料時發生錯誤: {e}")
                continue

    
    def process_all_files(self, augment_count_per_file=3):
        """
        批次處理所有MP3檔案
        
        Args:
            augment_count_per_file: 每個原始檔案生成的擴增資料數量
        """
        print(f"開始處理資料夾: {self.input_dir}")
        
        # 獲取所有MP3檔案
        mp3_files = list(self.input_dir.glob("*.mp3"))
        
        if not mp3_files:
            print(f"在 {self.input_dir} 中找不到MP3檔案")
            return
        
        print(f"找到 {len(mp3_files)} 個MP3檔案")
        
        # 處理每個檔案
        for idx, mp3_file in enumerate(mp3_files, 1):
            speaker_id = idx  # 使用檔案順序作為說話者ID
            print(f"\n=== 處理說話者 {speaker_id} ===")
            self.process_single_file(mp3_file, speaker_id, augment_count_per_file)
        
        print(f"\n完成所有檔案處理！輸出資料夾: {self.output_dir}")
        
        # 統計結果
        self.print_statistics()
    
    def print_statistics(self):
        """列印處理統計結果"""
        print("\n=== 處理統計 ===")
        
        total_files = 0
        for speaker_dir in self.output_dir.glob("speaker_*"):
            wav_files = list(speaker_dir.glob("*.wav"))
            speaker_id = speaker_dir.name.split("_")[1]
            print(f"說話者 {speaker_id}: {len(wav_files)} 個檔案")
            total_files += len(wav_files)
        
        print(f"總共生成: {total_files} 個音訊檔案")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='語音資料擴增批次處理程式')
    parser.add_argument('--input_dir', default=r'C:\Users\h_0212\專題\原始mp3', 
                       help='輸入MP3檔案資料夾路徑')
    parser.add_argument('--output_dir', default=r'C:\Users\h_0212\專題\擴增後資料vscode', 
                       help='輸出資料夾路徑')
    parser.add_argument('--augment_count', type=int, default=2, 
                       help='每個原始檔案生成的擴增資料數量')
    parser.add_argument('--sample_rate', type=int, default=16000, 
                       help='音訊取樣率')
    
    args = parser.parse_args()
    
    print("語音資料擴增批次處理程式")
    print("=" * 50)
    print(f"輸入資料夾: {args.input_dir}")
    print(f"輸出資料夾: {args.output_dir}")
    print(f"每個檔案擴增數量: {args.augment_count}")
    print(f"取樣率: {args.sample_rate} Hz")
    print("=" * 50)
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(args.input_dir):
        print(f"錯誤: 輸入資料夾不存在: {args.input_dir}")
        return
    
    # 創建擴增器並執行處理
    augmenter = VoiceDataAugmenter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
    
    try:
        augmenter.process_all_files(args.augment_count)
        print("\n處理完成！")
    except KeyboardInterrupt:
        print("\n使用者中斷處理")
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {e}")


if __name__ == "__main__":
    main()