import os
import numpy as np
import tensorflow as tf
import librosa
from pyftg.models.audio_data import AudioData
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_audio_data(audio_data_path, labels):
    data = []
    targets = []
    
    # 打印调试信息
    print(f"音频数据路径: {audio_data_path}")
    print(f"文件列表: {os.listdir(audio_data_path)}")

    for file_name in os.listdir(audio_data_path):
        if file_name.endswith(".wav"):  # 文件格式
            
            file_path = os.path.join(audio_data_path, file_name)
            print(f"正在处理文件: {file_path}")
            # 使用librosa加载音频文件
            y, sr = librosa.load(file_path, sr=None)

            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = np.mean(mfcc.T, axis=0)
            
            data.append(mfcc)
            
            if "_" not in file_name:
                label_key = file_name.replace(".wav", "")
                
            else:
                # 假设标签在文件名中，例如 "attack_001.wav"
                label_key = labels[file_name.split('_')[0]]
                
            label = labels.get(label_key, labels["unknown"])
            targets.append(label)
            
    if not data:
        raise ValueError("没有找到任何音频数据文件。请检查音频数据路径和文件格式。")
    
    data = np.array(data)
    targets = to_categorical(targets, num_classes=len(labels))
    
    return train_test_split(data, targets, test_size=0.2)

if __name__ == "__main__":
    #audio_data_path = "D:\\FightingICE\\Generative-Sound-AI-main\\data\\sounds"
    audio_data_path = "Generative-Sound-AI-main\\data\\sounds"
    # 更新标签字典以包含更多标签
    labels = {
        "AIR": 0, "BACK": 1, "STAND": 2, "STAND_D_DF_FA": 3, "CROUCH": 4,
        "DASH": 5, "EnergyCharge": 6, "FORWARD": 7, "Heartbeat": 8,
        "Hit": 9, "JUMP": 10, "LANDING": 11, "BGM0": 12, "THROW": 13,
        "WeakGuard": 14, "FOR":15, "unknown": 16
    }
    X_train, X_test, y_train, y_test = load_audio_data(audio_data_path, labels)
    np.savez("processed_data2.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("数据处理完成并保存！")
