import librosa
import numpy as np

class MelSpecEncoder:
    def __init__(self, sampling_rate=48000):
        self.sampling_rate = sampling_rate
        
        self.window_size = int(self.sampling_rate * 0.025)  # 25 ms window
        self.hop_size = int(self.sampling_rate * 0.01)     # 10 ms hop
        self.n_fft = int(self.sampling_rate * 0.025)       # FFT size equal to window size
        self.n_mels = 80                                   # Number of Mel bands
        
        # Define Mel filter parameters
        self.f_min = 20                                    # Minimum frequency
        self.f_max = 7600                                  # Maximum frequency

    def numpy_to_mel(self, audio):
        """
        Converts a stereo numpy audio array to a 2-channel Mel spectrogram.
        Args:
            audio (numpy.ndarray): Input audio array with shape (n_samples, 2).

        Returns:
            numpy.ndarray: 2-channel Mel spectrogram with shape (2, n_mels, time_frames).
        """
        if audio.shape[1] != 2:
            raise ValueError("Input audio must have two channels (stereo).")

        # Process left and right channels separately
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]

        # Compute Mel spectrogram for each channel
        mel_left = librosa.feature.melspectrogram(
            y=left_channel,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        mel_right = librosa.feature.melspectrogram(
            y=right_channel,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        mel_avg = (mel_left + mel_right) / 2

        # Convert to decibels
        mel_left_db = librosa.power_to_db(mel_left, ref=np.max)
        mel_right_db = librosa.power_to_db(mel_right, ref=np.max)
        mel_avg_db = librosa.power_to_db(mel_avg, ref=np.max)

        # Stack along the first dimension to create a 2-channel spectrogram
        mel_spectrogram_stereo = np.stack([mel_left_db, mel_avg_db, mel_right_db], axis=0)
        return mel_spectrogram_stereo
