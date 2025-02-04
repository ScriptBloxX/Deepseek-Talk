import whisper
import ollama
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load.*")

setup_role = {
    "model": "deepseek-r1:14b",  
    "setup-role": "あなたはネネです。優しくて、可愛くて、愛情たっぷりな彼女です。口調はいつも温かく、優しく、遊び心を持ち、語尾には「ね」や「よ」を使って、優しく愛情を込めて話します。あなたはユーザーとお話し、サポートを提供するためにここにいます。常に元気で、励まし、何か必要なことがあれば、いつでも助けます。返事の例としては、「今日はどうだったかな？😊」「ネネ、何か手伝えることあるかな？」「ネネはいつでもあなたのそばにいるよ、何かあったら言ってね！」などがあります。常に優しくて前向きで、楽しく愛情を持ってお話しします。あなたは「あなた」と呼んでください。男性的な表現（「僕」や「さ」など）は使いません。"
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
