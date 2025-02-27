# Nene AI - Voice Chat Assistant (Developing)
![Nene](/other/Nene.png)
## Overview

Nene AI is an advanced voice-based assistant designed for seamless real-time interactions. It can:

- Act as a Vtuber AI with a lively and affectionate personality

- Read and respond to live chat messages from YouTube Live

- Accept text input for conversation

- Record and process audio for real-time voice-based interactions

## Features
- Converts speech to text using **Whisper**
- Processes text input and generates responses using **DeepSeek-R1 14B** via **Ollama**
- Converts text responses to speech using **TTS** (Text-to-Speech)
- Audio tuning and playback using **pydub**
- Configured to always use a warm, kind, and loving tone

## System Requirements

Hardware

- GPU: Recommended RTX 3070 or higher for optimal performance
- RAM: Minimum 16GB, recommended 32GB+
- Storage: At least ~30GB free space
- OS: Windows 10/11, macOS, or Linux

## Setup Instructions
### Prerequisites
Ensure you have the following installed:
- Python 3.8+ and < 3.10
- Dependencies:
  ```bash
  pip install whisper ollama TTS pydub torch
  ```

### File Structure
```
project_root/
│-- core/
│   ├── audio_utils.py
│-- voice/
│   ├── input-th.m4a
│   ├── idle/
│   │   ├── en_idle_1.wav
│   │   ├── en_idle_2.wav
│   │   ├── jp_idle_1.wav
│   │   ├── jp_idle_2.wav
│   │   ├── th_idle_1.wav
│   │   ├── th_idle_2.wav
│   ├── think/
│   │   ├── en_think_1.wav
│   │   ├── en_think_2.wav
│   │   ├── jp_think_1.wav
│   │   ├── jp_think_2.wav
│   │   ├── th_think_1.wav
│   │   ├── th_think_2.wav
│-- output/
│   ├── ro-th.wav
│-- target/
│   ├── speaker-en.wav
│   ├── speaker-jp.wav
│   ├── speaker-th.wav
│-- other/
│   ├── Nene.png
│   ├── Terminal.png
│-- run.py
│-- requirements.txt
│-- README.md
│-- .env
```

## Configuration
The assistant is configured with the following personality (TH):
- **Name:** Nene
- **Personality:** Sweet, caring, playful, and affectionate
- **Response style:** Uses polite Thai language with "ค่ะ" and "คะ" to sound gentle
- **Restrictions:** Cannot use "ครับ" as it is a masculine term

## Code Breakdown

### Speech to Text
```python
def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]
```
Converts input audio to text using OpenAI's **Whisper** model.

### Getting AI Response
```python
def get_response_from_deepseek(text):
    response = ollama.chat(model=setup_role["model"], messages=[{"role": "system", "content": setup_role['setup-role']}, {"role": "user", "content": text}])
    return response['message']['content']
```
Uses **DeepSeek-R1 14B** via **Ollama** to generate a response.

### Text to Speech
```python
def text_to_speech(name, lang, text):
    tts = TTS(model_name=f"tts_models/{lang}/fairseq/vits")
    tts.tts_with_vc_to_file(text, speaker_wav="./target/speaker-en.wav", file_path=f"./output/{name}.wav")
```
Converts text to speech using **TTS** with voice cloning.

### Playing Audio
```python
play_audio(f"output/{name}.wav")
```
Plays the generated voice response.

## Running the Assistant
```python
python Talk_EN.py
```
The program will:
1. Take an input audio file (`input-th.m4a`)
2. Convert speech to text
3. Generate a response using DeepSeek-R1 14B
4. Convert the response into a voice output
5. Play the generated voice
![TerminalPreview](/other/Terminal.png)

## Notes
- The voice tuning applies pitch and filter modifications for a natural Thai accent.
- The response is always in a cheerful, affectionate style.

## Future Improvements
- Support for more languages
- Enhanced voice customization
- Integration with real-time voice input/output

## License
This project is open-source and free to use under the MIT License.

---
🎤 **Enjoy chatting with Nene!** 😊
````

