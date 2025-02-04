import pyaudio
import wave

def play_audio(file_path):
    wf = wave.open(file_path, 'rb')

    p = pyaudio.PyAudio()

    stream_speakers = p.open(format=pyaudio.paInt16,
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

    stream_mic = p.open(format=pyaudio.paInt16,
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=8)

    data = wf.readframes(1024)

    while data:
        stream_speakers.write(data)

        stream_mic.write(data)

        data = wf.readframes(1024)

    stream_speakers.stop_stream()
    stream_speakers.close()

    stream_mic.stop_stream()
    stream_mic.close()

    p.terminate()
