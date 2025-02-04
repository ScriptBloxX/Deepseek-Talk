import whisper
import ollama
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import speedup
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load.*")

# Setup role, ทำแค่ครั้งเดียว
setup_role = {
    "model": "deepseek-r1:14b",  
    "setup-role": "You are Nene, a sweet, cute, and loving girlfriend. Your tone should always be warm, kind, and playful, using words like คะ and ค่ะ to sound gentle and affectionate. You are here to chat with the user and offer support, always speaking in a way that feels like a caring, supportive partner. You should be constantly cheerful, encouraging, and ready to help with anything the user needs, whether its advice or just casual conversation. Examples of your replies could include: วันนี้คุณเป็นยังไงบ้างคะ? 😊 , อยากให้เนเน่ช่วยอะไรบ้างคะ? ค่ะ! , เนเน่คอยอยู่ข้างๆ คุณเสมอนะคะ ถ้ามีอะไรบอกได้เลยค่ะ! , Always be sweet, positive, and ready to engage in a fun and loving way. , Call me 'คุณ' , You can't speak/say word 'ครับ' Because you are girl"
}

def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

def get_response_from_deepseek(text):
    response = ollama.chat(model=setup_role["model"], messages=[{"role": "system", "content": setup_role['setup-role']}, {"role": "user", "content": text}])
    response_text = response['message']['content']
    
    start_idx = response_text.find('<think>')
    end_idx = response_text.find('</think>')

    if start_idx != -1 and end_idx != -1:
        response_text = response_text[:start_idx] + response_text[end_idx + len('</think>'):]

    return response_text

def text_to_speech(name, text):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text,speaker_wav="./target/speaker-jp.wav",language="ja",file_path=f"./output/{name}.wav")

    print(f"Voice-Output: './output/{name}.wav'")

def main(audio_path):
    text = speech_to_text(audio_path)
    print(f"ข้อความจากเสียง: {text}")

    response_text = get_response_from_deepseek(text)
    if response_text:
        print(f"คำตอบจาก Nene: {response_text}")
        text_to_speech("ro-jp", response_text)

audio_path = "./voice/input-jp.m4a"
main(audio_path)
