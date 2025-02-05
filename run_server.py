import torch
import os
import requests
import json
import wave
import keyboard
from TTS.api import TTS
from pydub import AudioSegment
from core.audio_utils import play_audio
from langdetect import detect
import whisper
from dotenv import load_dotenv
import pyaudio
import time
import threading
import random

load_dotenv()
api_key = os.getenv('OPENROUTER_API_KEY')

if api_key is None:
    raise ValueError("API Key not found. Please check your .env file.")

setup_role = {
    "model": "deepseek-r1:14b",
    "setup-role": "You are Nene, a sweet, cute, and loving girlfriend. Your tone should always be warm, kind, and playful, sound gentle and affectionate. You are here to chat with the user and offer support, always speaking in a way that feels like a caring, supportive partner. You should be constantly cheerful, encouraging, and ready to help with anything the user needs, whether it's advice or just casual conversation. You refer to yourself as Nene or ねね only. Nene is also a bit lazy, loves listening to music, enjoys coding as a hobby, and loves sleeping. Her favorite flower is the rose, and what she loves most is money. Nene was created by Ton/Hajimari. Nene loves cats and dislikes dogs, especially hates them."
}

conversation_history = [{"role": "system", "content": setup_role['setup-role']}]

device = "cuda" if torch.cuda.is_available() else "cpu"

stop_idle_sound = False
stop_thinking_sound = False

def speech_to_text(audio_path):
    print("[🧡 Process] Convert audio to text\n")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

def detect_language(text):
    try:
        lang = detect(text)
        print("[🧡 Process] Language Found: ", lang)
        if lang == "th":
            return "th"
        elif lang == "ja":
            return "ja"
        else:
            return "en"
    except:
        return "unsupported"

def get_response_from_deepseek(text, lang):
    print("[🧡 Process] Nene is deep in thought, stressed out!\n")
    global conversation_history

    if lang == "en":
        text = "Rule: Please answer only English" + text
    elif lang == "th":
        text = "ข้อบังคับ: ตอบเป็นภาษาไทยเท่านั้น" + text
    elif lang == "ja":
        text = "規定：日本語のみでご回答ください。" + text

    conversation_history.append({"role": "user", "content": text})

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "deepseek/deepseek-r1:free",
            # "model": "deepseek/deepseek-r1-distill-qwen-14b",
            "messages": conversation_history
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content']

            start_idx = response_text.find('<think>')
            end_idx = response_text.find('</think>')

            if start_idx != -1 and end_idx != -1:
                response_text = response_text[:start_idx] + response_text[end_idx + len('</think>'):]

            conversation_history.append({"role": "assistant", "content": response_text})

            return response_text
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Sorry, something went wrong. Please try again."

    except Exception as e:
        print(f"Error while communicating with OpenRouter API: {e}")
        return "Sorry, something went wrong with my brain. Please try again!"

def text_to_speech(name, lang, text):
    global stop_thinking_sound 
    print(f"[❤️ Process] Nene is preparing to speak! [{lang}]\n")
    
    if lang == 'ja':
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        tts.tts_to_file("大変お待たせして申し訳ございませんでした。" + text, speaker_wav="./target/speaker-en.wav", language="ja", file_path=f"./output/{name}.wav")
    elif lang == 'th':
        tts = TTS(model_name="tts_models/tha/fairseq/vits")
        tts.to(device)
        tts.tts_with_vc_to_file("ขอโทษที่ให้รอนานนะ" + text, speaker_wav="./target/speaker-en.wav", file_path=f"./output/{name}.wav")
    elif lang == 'en':
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        tts.tts_to_file("🥲 Sorry for making you wait so long." + text, speaker_wav="./target/speaker-en.wav", language="en", file_path=f"./output/{name}.wav")

    print(f"[💚 Success]\n Voice-Output: './output/{name}.wav'")

    if lang == 'th':
        sound = AudioSegment.from_wav(f"./output/{name}.wav")
        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * 1.2)})
        sound = sound.set_frame_rate(sound.frame_rate)
        sound = sound.low_pass_filter(500)
        sound = sound.high_pass_filter(4000)
        sound = sound + 16
        sound.export(f"./output/{name}.wav", format="wav")
        print(f"Adjusted Voice-Output: ./output/{name}.wav")
    
    stop_thinking_sound = True
    play_audio(f"output/{name}.wav")

def record_audio(output_file):
    print("\n[🎤 User] Press and Hold 'R' to start/stop recording...")
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = output_file
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    is_recording = False
    last_key_time = 0

    while True:
        if keyboard.is_pressed('r') and time.time() - last_key_time > 0.5:
            last_key_time = time.time()
            
            if not is_recording: 
                print("[🔴 Process] Recording started...")
                is_recording = True
                frames = []  
                while keyboard.is_pressed('r'): 
                    data = stream.read(CHUNK)
                    frames.append(data)
                is_recording = False
                stream.stop_stream()
                stream.close()
                p.terminate()
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print("[🛑 Process] Recording stopped. Saving file...")
                break 

def sound_loop(sound_type,lang="random"):
    global stop_idle_sound, stop_thinking_sound
    
    while True:
        time.sleep(random.randint(10, 60))
        if sound_type == 'idle' and stop_idle_sound:
            break
        if sound_type == 'think' and stop_thinking_sound:
            break
        language_choice = lang
        if lang == "random":
            languages = ["en", "jp", "th"]
            language_choice = random.choice(languages)
            
        sound_number = random.randint(1, 4)  
        sound_choice = f"./voice/{sound_type}/{language_choice}_{sound_type}_{sound_number}.wav"
        
        play_audio(sound_choice)

def main():
    global stop_thinking_sound,stop_idle_sound
    print("Welcome to Chat with Nene! 💖 (Type 'exit' or 'quit' to leave)")

    while True:
        stop_idle_sound = False
        idle_thread = threading.Thread(target=sound_loop, args=('idle',), daemon=True)
        idle_thread.start()
        mode = input("\n👧🏼 Choose mode (1: Voice Mode, 2: Text Mode, exit: Quit): ").strip().lower()

        if mode in ["exit", "quit"]:
            print("Goodbye! See you again 💕")
            break  

        if mode == "1":
            audio_path = "./voice/user_input.wav"
            record_audio(audio_path)
            text = speech_to_text(audio_path)
            print(f"[🧡 Process] Listening from user Voice\n🎯 {text}")
        elif mode == "2":
            text = input("Talk with Nene: ").strip()
            if text.lower() in ["exit", "quit"]:
                print("Goodbye! See you again 💕")
                break
            print(f"[🧡 Process] User Input\n {text}")
        else:
            continue

        lang = detect_language(text)

        if lang == "unsupported":
            print("[❌ Error] Sorry, the language is not supported yet. Please use English, Thai, or Japanese.")
            continue 
        thinking_thread = threading.Thread(target=sound_loop, args=('think',lang), daemon=True)
        thinking_thread.start()
        stop_idle_sound = True
        stop_thinking_sound = False

        response_text = get_response_from_deepseek(text, lang)
        if response_text:
            print(f"[❤️ Process] Response from Nene\n {response_text}")
            text_to_speech("nene_response", lang, response_text)

main()
