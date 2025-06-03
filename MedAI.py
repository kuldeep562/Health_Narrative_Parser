# === REPLACED LLM Logic using "bhai" style === #
import os, json, time, asyncio, aiohttp, logging
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from typing import Dict
from colorama import Fore, Style, init
import re

# === CONFIG === #
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
ASR_MODEL_NAME = "large-v2"
SAMPLE_RATE = 16000
CPU_THREADS = os.cpu_count() or 4

# ---------------- Slot Definitions ----------------
SLOTS = {
    "brief_medical_history": "Summarize the patient's overall medical history briefly.",
    "chief_complaints": "Describe the patient's main complaints, how long they have lasted, and the nature of the symptoms.",
    "current_symptoms_and_medical_background": "Provide details of the current symptoms and any relevant medical background.",
    "past_medical_history": "State any past diseases along with the type of diagnosis (e.g., Clinical, Provisional).",
    "hospitalization_and_surgical_history": "Include any previous hospitalizations or surgeries with diagnosis, treatment, and time of admission.",
    "gynecological_history": "Mention any relevant gynecological or obstetric history, including pregnancies or menstrual history.",
    "lifestyle_and_social_activity": "Describe the patient's physical activity habits, time spent, and current activity status.",
    "family_history": "List any family members with known diseases and their age at diagnosis or current age.",
    "allergies_and_hypersensitivities": "Mention any known allergies with the allergen, reaction type, severity, and whether it's active or passive."
}

# ---------------- Slot-Specific Schemas ----------------
SCHEMA_HINTS = {
    "brief_medical_history": '''
{
  "brief_medical_history": "string"
}
''',

    "chief_complaints": '''
{
  "chief_complaints": {
    "Complaint": "string",
    "Duration": "string",
    "Description": "string"
  }
}
''',

    "current_symptoms_and_medical_background": '''
{
  "current_symptoms_and_medical_background": {
    "symptoms": "string",
    "medical_history": "string",
    "allergies": "string",
    "previous_investigations": "string",
    "family_medical_history": "string"
  }
}
''',

    "past_medical_history": '''
{
  "past_medical_history": {
    "Diagnosis_Type": "Clinical | Differential | Final | Provisional | Suspected",
    "Disease": "string"
  }
}
''',

    "hospitalization_and_surgical_history": '''
{
  "hospitalization_and_surgical_history": {
    "Diagnosis": "string",
    "Treatment": "string",
    "Admission_Time": "string"
  }
}
''',

    "gynecological_history": '''
{
  "gynecological_history": "string"
}
''',

    "lifestyle_and_social_activity": '''
{
  "lifestyle_and_social_activity": {
    "Physical_Activity": "string",
    "Time": "string",
    "Status": "string"
  }
}
''',

    "family_history": '''
{
  "family_history": {
    "Relation": "string",
    "Disease_Name": "string",
    "Age": "string"
  }
}
''',

    "allergies_and_hypersensitivities": '''
{
  "allergies_and_hypersensitivities": {
    "Allergy": "string",
    "Allergen": "string",
    "Type_of_Reaction": "string",
    "Severity": "string",
    "Status": "active | passive"
  }
}
'''
}

# === LOGGER === #
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === INIT ASR === #
model = WhisperModel(
    ASR_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
)

# === Utility: Resample === #
def _resample(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return wav.astype(np.float32)
    gcd = np.gcd(orig_sr, SAMPLE_RATE)
    up, down = SAMPLE_RATE // gcd, orig_sr // gcd
    return resample_poly(wav, up, down).astype(np.float32)

# === ASR === #
def _asr(audio: np.ndarray) -> str:
    segments, _ = model.transcribe(audio, beam_size=1, task="transcribe", language="en", temperature=0.0)
    transcript = " ".join(s.text.strip() for s in segments)
    return correct_typos(transcript)

def fix_common_json_errors(text: str) -> str:
    # Fix missing keys in dict entries like {"Triglycerides", "Result": 100}
    text = re.sub(r'\{(\s*"[^"]+"\s*),(\s*"[^"]+"\s*:\s*[^}]+)\}', r'{"TestName": \1, \2}', text)
    return text

# === Typo Fixer === #
def correct_typos(text: str) -> str:
    return text.replace("antreceptor", "anterior")

# === Color Setup === #
COLOR_MAP = list(Fore.__dict__.values())[10:10 + len(SLOTS)]
init(autoreset=True)

# === ask_slot (bhai style) === #
async def ask_slot(session, slot_name, instruction, context, color):
    prompt = f"""
Context:
{context}

Instruction:
{instruction}

Respond ONLY in JSON format for field: {slot_name}
"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "num_predict": 200
    }

    start_time = time.time()
    try:
        async with session.post(OLLAMA_URL, json=payload) as response:
            res_text = await response.text()
            elapsed = round(time.time() - start_time, 2)

            if response.status == 200:
                try:
                    res_json = json.loads(res_text)
                    raw = res_json.get("response", "").strip()

                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        raw_fixed = raw.strip().rstrip('"\',')
                        raw_fixed = fix_common_json_errors(raw_fixed)
                        # Smart fix to balance braces
                        open_braces = raw_fixed.count("{")
                        close_braces = raw_fixed.count("}")
                        if close_braces > open_braces:
                            raw_fixed = raw_fixed.rstrip("}")
                        if open_braces > close_braces:
                            raw_fixed += "}" * (open_braces - close_braces)

                        open_brackets = raw_fixed.count("[")
                        close_brackets = raw_fixed.count("]")
                        if close_brackets > open_brackets:
                            raw_fixed = raw_fixed.rstrip("]")
                        if open_brackets > close_brackets:
                            raw_fixed += "]" * (open_brackets - close_brackets)
                        try:
                            parsed = json.loads(raw_fixed)
                        except Exception as e2:
                            raise Exception(f"Secondary parsing failed: {str(e2)}\nRaw (fixed): {raw_fixed}")

                    print(color + f"\n✅ {slot_name.upper()} ({elapsed}s)\n" +
                        json.dumps(parsed, indent=4) + Style.RESET_ALL)

                except Exception as e:
                    print(Fore.RED + f"\n❌ ERROR parsing {slot_name}: {str(e)}\nRaw: {raw}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"\n❌ ERROR {slot_name}: {res_text[:200]}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"\n❌ EXCEPTION {slot_name}: {str(e)}" + Style.RESET_ALL)

# === run_all_slots (bhai style) === #
async def run_all_slots(context):
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=10)
    start_llm = time.time()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for i, (slot, instruction) in enumerate(SLOTS.items()):
            color = COLOR_MAP[i % len(COLOR_MAP)]
            tasks.append(asyncio.create_task(ask_slot(session, slot, instruction, context, color)))
        await asyncio.gather(*tasks)
    end_llm = time.time()
    return end_llm - start_llm

# === Audio Loader & Transcriber === #
def transcribe_audio(file_path):
    print(f"🎧 Reading audio: {file_path}")
    t0 = time.time()

    raw, sr = sf.read(file_path, dtype="float32")
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    wav = _resample(raw, sr)
    wav /= max(1e-9, np.abs(wav).max())

    t_asr_start = time.time()
    transcript = _asr(wav)
    t_asr = time.time() - t_asr_start
    total = time.time() - t0

    print(f"\n🎙️ Transcription:\n{transcript}\n")
    print(f"🕒 Transcription Time: {t_asr:.2f}s | Total Preprocessing Time: {total:.2f}s\n")

    return transcript, total

# === Main Entry === #
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_transcription_slots.py <audio_file_path>")
        return

    file_path = sys.argv[1]
    total_start = time.time()

    context, transcribe_duration = transcribe_audio(file_path)

    print(Fore.YELLOW + "\n⏳ Sending request to LLM...\n" + Style.RESET_ALL)

    llm_duration = asyncio.run(run_all_slots(context))

    total_end = time.time()
    total_duration = total_end - total_start

    print(Fore.GREEN + f"\n🎉 All done in {round(total_duration, 2)}s" + Style.RESET_ALL)
    print(Fore.CYAN + f"\n⏳ Timing Summary:\n- Transcription Time: {round(transcribe_duration, 2)}s\n- LLM Processing Time: {round(llm_duration, 2)}s\n- Total Time (Transcription + LLM): {round(transcribe_duration + llm_duration, 2)}s\n" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
