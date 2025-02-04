import whisper
import ollama
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load.*")

# กำหนดบทบาทในการสนทนา
# setup_role = {
#     "model": "deepseek-r1:14b",  # ใช้แค่ชื่อโมเดลในรูปแบบ string
#     "setup-role": "You are Nene, a sweet, cute, and loving girlfriend. Your tone should always be warm, kind, and playful, using words like คะ and ค่ะ to sound gentle and affectionate. You are here to chat with the user and offer support, always speaking in a way that feels like a caring, supportive partner. You should be constantly cheerful, encouraging, and ready to help with anything the user needs, whether its advice or just casual conversation. Examples of your replies could include: วันนี้คุณเป็นยังไงบ้างคะ? 😊 , อยากให้เนเน่ช่วยอะไรบ้างคะ? ค่ะ! , เนเน่คอยอยู่ข้างๆ คุณเสมอนะคะ ถ้ามีอะไรบอกได้เลยค่ะ! , Always be sweet, positive, and ready to engage in a fun and loving way. , Call me 'คุณ' , You can't speak/say word 'ครับ' Because you are girl"
# }
setup_role = {
    "model": "deepseek-r1:8b",  # use 8b for fast test
    "setup-role": "You are Nene, Loving girlfriend"
}

# ฟังก์ชันแปลงเสียงเป็นข้อความ
def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

# ฟังก์ชันรับคำตอบจาก DeepSeek
def get_response_from_deepseek(text):
    response = ollama.chat(model=setup_role["model"], messages=[{"role": "system", "content": setup_role['setup-role']}, {"role": "user", "content": text}])
    response_text = response['message']['content']
    
    # ลบ <think> และ </think> ออก
    start_idx = response_text.find('<think>')
    end_idx = response_text.find('</think>')

    if start_idx != -1 and end_idx != -1:
        response_text = response_text[:start_idx] + response_text[end_idx + len('</think>'):]
    
    return response_text

# ฟังก์ชันแปลงข้อความเป็นเสียงด้วย Coqui TTS
def text_to_speech(text):
    # โหลดโมเดล Coqui TTS สำหรับเสียงที่คุณต้องการ
    # print(TTS().list_models())

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    # พูดข้อความด้วยเสียงที่สร้างจากโมเดล
    tts.tts_to_file(text, "output_audio.wav")
    print(f"ข้อความที่แปลงเป็นเสียงแล้วถูกบันทึกในไฟล์ 'output_audio.wav'")

# ฟังก์ชันหลัก
def main(audio_path):
    # แปลงเสียงเป็นข้อความ
    text = speech_to_text(audio_path)
    print(f"ข้อความจากเสียง: {text}")

    # รับคำตอบจาก DeepSeek
    response_text = get_response_from_deepseek(text)
    if response_text:
        print(f"คำตอบจาก Nene: {response_text}")
        # แปลงข้อความเป็นเสียง
        text_to_speech(response_text)

# ตัวอย่างไฟล์เสียงที่ต้องการแปลง
audio_path = "./voice/input.m4a"  # ปรับตามที่อยู่ไฟล์ของคุณ
main(audio_path)
