import tkinter as tk
from tkinter import filedialog
from tkinter import *
import pyaudio
import wave
import numpy as np
from tensorflow.keras.models import model_from_json
from scipy.io.wavfile import read

class VoiceDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry('400x200')
        self.master.title('Voice Detector')
        self.master.configure(background='#CDCDCD')

        self.label_result = Label(self.master, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.label_result.pack(pady=20)

        self.model = self.load_model("voice_detection_model.json", "voice_detection_model.h5")

        self.setup_gui()

    def setup_gui(self):
        record_button = Button(self.master, text="Record Voice", command=self.record_voice, padx=10, pady=5)
        record_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        record_button.pack(pady=20)

    def record_voice(self):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = 44100  # Record at 44100 samples per second
        seconds = 5

        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        print("Recording...")

        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded audio as a WAV file
        wav_file = "recorded_audio.wav"
        wf = wave.open(wav_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Perform voice detection using your model
        result = self.detect_voice(wav_file)

        self.label_result.configure(foreground="#011638", text=result)

    def detect_voice(self, audio_file):
        try:
            rate, audio_data = read(audio_file)
            audio_data = np.array(audio_data, dtype=float)

            # Preprocess audio data (you may need to customize this based on your model requirements)
            # For example, extract features or reshape the data to match the input shape of your model

            # Use your voice detection model to make predictions
            prediction = self.model.predict(np.expand_dims(audio_data, axis=0))

            # Adjust the threshold as needed
            threshold = 0.5
            result = "Voice Detected" if prediction > threshold else "No Voice Detected"

            return result
        except Exception as e:
            print(f"Error detecting voice: {e}")
            return "Error"

    def load_model(self, voice, voice_detection_model):
        try:
            with open(voice, "r") as file:
                loaded_model_json = file.read()
                model = model_from_json(loaded_model_json)

            model.load_weights(voice_detection_model)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            return model
        except Exception as e:
            print(f"Error loading voice detection model: {e}")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceDetectorApp(root)
    root.mainloop()
