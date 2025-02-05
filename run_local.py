import torch
import whisper
import ollama
from TTS.api import TTS
from pydub import AudioSegment
from core.audio_utils import play_audio
import warnings
from langdetect import detect
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load.*")

setup_role = {
    "model": "deepseek-r1:14b",
    "setup-role": "You are Nene, a sweet, cute, and loving girlfriend. Your tone should always be warm, kind, and playful, sound gentle and affectionate. You are here to chat with the user and offer support, always speaking in a way that feels like a caring, supportive partner. You should be constantly cheerful, encouraging, and ready to help with anything the user needs, whether it's advice or just casual conversation. You refer to yourself as Nene or ねね only. Nene is also a bit lazy, loves listening to music, enjoys coding as a hobby, and loves sleeping. Her favorite flower is the rose, and what she loves most is money. Nene was created by Ton/Hajimari. Nene loves cats and dislikes dogs, especially hates them."
    }

conversation_history = [{"role": "system", "content": setup_role['setup-role']}]

device = "cuda" if torch.cuda.is_available() else "cpu"

def speech_to_text(audio_path):
    print("[🧡 Process] Convert audio to text\n")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

def detect_language(text):
    try:
        lang = detect(text)
        print("[🧡 Process] Language Found: ",lang)
        if lang == "th":
            return "th"
        elif lang == "ja":
            return "ja"
        else:
            return "en"
    except:
        return "unsupported"

def get_response_from_deepseek(text,lang):
    print("[🧡 Process] Nene is deep in thought, stressed out!\n")
    global conversation_history

    if lang == "en":
        text = "Rule: Please answer only English"+text
    elif lang == "th":
        text = "ข้อบังคับ: ตอบเป็นภาษาไทยเท่านั้น"+text
    elif lang == "ja":
        text = "規定：日本語のみでご回答ください。"+text

    conversation_history.append({"role": "user", "content": text})

    response = ollama.chat(model=setup_role["model"], messages=conversation_history)
    response_text = response['message']['content']

    start_idx = response_text.find('<think>')
    end_idx = response_text.find('</think>')

    if start_idx != -1 and end_idx != -1:
        response_text = response_text[:start_idx] + response_text[end_idx + len('</think>'):]

    conversation_history.append({"role": "assistant", "content": response_text})

    return response_text

def text_to_speech(name, lang, text):
    print(f"[❤️ Process] Nene is preparing to speak! [{lang}]\n")
    
    if lang == 'ja':
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        tts.tts_to_file("大変お待たせして申し訳ございませんでした。"+text, speaker_wav="./target/speaker-en.wav", language="ja", file_path=f"./output/{name}.wav")
    elif lang == 'th':
        tts = TTS(model_name="tts_models/tha/fairseq/vits")
        tts.to(device)
        # i use speaker-en because i think the voice is better than speaker-th
        tts.tts_with_vc_to_file("ขอโทษที่ให้รอนานนะ"+text, speaker_wav="./target/speaker-en.wav", file_path=f"./output/{name}.wav")
    elif lang == 'en':
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        tts.tts_to_file("🥲 Sorry for making you wait so long."+text, speaker_wav="./target/speaker-en.wav", language="en", file_path=f"./output/{name}.wav")

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
    
    play_audio(f"output/{name}.wav")

def main():
    print("Welcome to Chat with Nene! 💖 (Type 'exit' or 'quit' to leave)")

    while True:
        mode = input("\nChoose mode (1: Voice Mode, 2: Text Mode, exit: Quit): ").strip().lower()

        if mode in ["exit", "quit"]:
            print("Goodbye! See you again 💕")
            break

        if mode == "1":
            audio_path = input("Input your audio path file (eg ./voice/input-en.m4a): ").strip()
            text = speech_to_text(audio_path)
            print(f"[🧡 Process] Listening from user Voice\n {text}")
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
            print("Sorry, the language is not supported yet. Please use English, Thai, or Japanese.")
            continue

        response_text = get_response_from_deepseek(text,lang)
        if response_text:
            print(f"[❤️ Process] Response from Nene {response_text}")
            text_to_speech("nene_response", lang, response_text)

main()
