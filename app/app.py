import os
LOCAL_TEMP = os.path.join(os.getcwd(), "gradio_temp")
os.makedirs(LOCAL_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = LOCAL_TEMP
import re
import gc
import time
import subprocess
import shutil
import tempfile
import warnings
from num2words import num2words
from decimal import Decimal

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# --- Download NLTK tokenizers if missing ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK tokenizers...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Zero Dependency Extractors & Tools ---
import fitz  # PyMuPDF
import mobi
from ebooklib import epub, ITEM_DOCUMENT
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# --- OmniVoice Engine Imports ---
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
print(f"Using Device: {device}")

# --- Load OmniVoice Model ---
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
print(f"Loading OmniVoice model from {CHECKPOINT} to {device} ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map=device,
    load_asr=True,
    torch_dtype=torch.float16  
)
sampling_rate = model.sampling_rate
print("OmniVoice Model loaded successfully!")

class EbookProgressUpdater:
    def __init__(self, progress, num_total_chunks, ebook_idx, num_ebooks, start_time):
        self._progress = progress
        self.num_total_chunks = num_total_chunks
        self.ebook_idx = ebook_idx
        self.num_ebooks = num_ebooks
        self.start_time = start_time
        self.current_chunk_idx = 0

    def set_chunk_index(self, i):
        self.current_chunk_idx = i

    def __call__(self, value, desc=None):
        chunk_percent = 0.0
        if desc:
            match = re.search(r'(\d+\.?\d*)%', desc)
            if match:
                try: chunk_percent = float(match.group(1))
                except ValueError: chunk_percent = 0.0

        book_progress_fraction = (self.current_chunk_idx + (chunk_percent / 100.0)) / self.num_total_chunks
        book_progress_percent = book_progress_fraction * 100
        elapsed_seconds = time.time() - self.start_time
        
        elapsed_hours = int(elapsed_seconds // 3600)
        elapsed_minutes = int((elapsed_seconds % 3600) // 60)
        elapsed_secs = int(elapsed_seconds % 60)
        elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}"

        etr_str = "Calculating..."
        if elapsed_seconds > 2 and book_progress_fraction > 0.001:
            try:
                total_estimated_time = elapsed_seconds / book_progress_fraction
                etr_seconds = max(0, total_estimated_time - elapsed_seconds)
                etr_hours = int(etr_seconds // 3600)
                etr_minutes = int((etr_seconds % 3600) // 60)
                etr_secs = int(etr_seconds % 60)
                etr_str = f"{etr_hours:02d}:{etr_minutes:02d}:{etr_secs:02d}"
            except ZeroDivisionError:
                 etr_str = "Calculating..."

        if self.num_ebooks > 1:
            final_desc = f"Current Book: {self.ebook_idx + 1}/{self.num_ebooks} ({book_progress_percent:.1f}%) | Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | Elapsed: {elapsed_str} | ETR: {etr_str}"
        else:
            final_desc = f"Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | Elapsed: {elapsed_str} | ETR: {etr_str}"

        self._progress(value, desc=final_desc)

def convert_numbers_to_words(txt):
    txt = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', lambda m: num2words(int(m.group(0)), to='year'), txt)
    txt = re.sub(r'(\d+)(st|nd|rd|th)\b', lambda m: num2words(int(m.group(1)), to='ordinal'), txt, flags=re.IGNORECASE)
    def number_replacer(match):
        num_str = match.group(0).replace(',', '')
        try: return num2words(Decimal(num_str))
        except: return num_str
    number_pattern = r'\b\d{1,3}(?:,\d{3})*\.\d+\b|\b\d{1,3}(?:,\d{3})*\b'
    txt = re.sub(number_pattern, number_replacer, txt)
    return txt

def strip_footnotes(text: str) -> str:
    """
    Cleans ebook text for TTS by removing mechanical citations and 
    footnote markers while preserving narrative content and asides.
    """
    
    # 1. Remove Bracketed Numeric Citations or [sic]
    # Examples: [1], [1, 2], [1-5], [sic], [ibid]
    # This is safe because authors rarely use square brackets for narrative.
    text = re.sub(r'\[\d+(?:,\s*\d+|-\d+)*\]', '', text)
    text = re.sub(r'\[(?:sic|ibid|ref|page\s\d+)\]', '', text, flags=re.IGNORECASE)

    # 2. Remove Asterisk/Symbol footnote markers
    # Matches: *1, *2, †, ‡ at the end of words or start of lines
    text = re.sub(r'(?<=\w)[\*\†\‡\§]\d*', '', text)
    text = re.sub(r'^\s*[\*\†\‡\§]\d*\s*', '', text, flags=re.MULTILINE)

    # 3. Handle trailing numeric footnotes (The "Word.12" or "Word12" problem)
    # Only removes numbers if they are 1-3 digits and attached to the end of a word/punctuation.
    # Logic: Look for a word, then a period, then a small number. 
    # This avoids touching "Version 2.0" or "8.8" because of the lookbehind.
    text = re.sub(r'(?<=[a-zA-Z])\.\d{1,3}\b', '.', text)
    
    # 4. Remove Author-Year Citations (Academic style)
    # Example: (Smith, 2020) or (Jones 1999)
    # This is safer than removing ALL parentheses.
    text = re.sub(r'\(\s*[A-Z][a-z]+,?\s+\d{4}\s*\)', '', text)

    # 5. Clean up "Orphaned" footnote numbers at the start of lines
    # Often found in OCR or poorly converted EPUBS.
    text = re.sub(r'^\s*\d+\b\s*', '', text, flags=re.MULTILINE)

    # 6. Final Polish: Remove double spaces and trailing whitespace
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def clean_and_normalize_text(raw_text: str) -> str:
    text = raw_text
    text = re.sub(r'\*\s*\d+\b', '', text) 
    text = re.sub(r'\[[^\]]*\]', '', text)  
    academic_terms = r'\b(?:spp?\.?|ssp\.?|subsp\.?|var\.?|f\.?|cf\.?|e\.g\.?|i\.e\.?|viz\.?|see|fig\.?|plate|chapter|probably)\b'
    year_pattern = r'\d{4}'
    combined_pattern = rf'\([^)]*(?:{academic_terms}|{year_pattern})[^)]*\)'
    text = re.sub(combined_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'#(\d)', r'number \1', text)     
    text = re.sub(r'#([a-zA-Z])', r'hashtag \1', text) 
    text = text.replace('#', ' pound ')               
    def currency_replacer(match):
        symbol = match.group(1)
        amount = match.group(2)
        mapping = {'$': 'dollar', '£': 'pound', '€': 'euro', '¥': 'yen'}
        curr_name = mapping.get(symbol, 'dollars')
        if amount != "1": curr_name += "s"
        return f"{amount} {curr_name}"
    text = re.sub(r'([$£€¥])(\d+(?:\.\d{2})?)', currency_replacer, text)
    symbol_map = {'—': ', ', '–': ', ', '&': ' and ', '%': ' percent ', '@': ' at ', 'µm': ' micrometers ', '°': ' degrees ', '+': ' plus ', '=': ' equals ', '/': ' or '}
    for old, new in symbol_map.items(): text = text.replace(old, new)
    text = re.sub(r'\bSt\.\s+(?=[A-Z])', 'Saint ', text)
    text = re.sub(r'(?<=\d)\s*St\.\b', ' Street', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=[A-Z][a-z])\s+St\.\b', ' Street', text)
    safe_abbreviations = {"Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Dr.": "Doctor", "Prof.": "Professor", "Rev.": "Reverend", "Hon.": "Honorable", "Jr.": "Junior", "Sr.": "Senior", "Gen.": "General", "Adm.": "Admiral", "Capt.": "Captain", "Cmdr.": "Commander", "Lt.": "Lieutenant", "Sgt.": "Sergeant", "Co.": "Company", "Corp.": "Corporation", "Inc.": "Incorporated", "Ltd.": "Limited", "LLC": "Limited Liability Company", "vs.": "versus", "et al.": "et alia", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is", "Ph.D.": "Doctor of Philosophy", "M.A.": "Master of Arts", "B.A.": "Bachelor of Arts", "pp.": "pages", "vol.": "volume", "U.S.": "United States", "U.S.A.": "United States of America", "U.K.": "United Kingdom", "E.U.": "European Union", "Ave.": "Avenue", "Blvd.": "Boulevard", "Rd.": "Road", "sq.": "square", "cu.": "cubic", "deg.": "degrees", "A.M.": "ay em", "P.M.": "pee em", "Jan.": "January", "Feb.": "February", "Mar.": "March", "Apr.": "April", "Jun.": "June", "Jul.": "July", "Aug.": "August", "Sep.": "September", "Oct.": "October", "Nov.": "November", "Dec.": "December", "approx.": "approximately", "dept.": "department", "apt.": "apartment", "est.": "established"}
    unit_abbreviations = {"mm": "millimeters", "cm": "centimeters", "m": "meters", "km": "kilometers", "mg": "milligrams", "g": "grams", "kg": "kilograms", "in": "inches", "ft": "feet", "yd": "yards", "mi": "miles", "oz": "ounces", "lb": "pounds", "lbs": "pounds", "mph": "miles per hour", "kph": "kilometers per hour"}
    for abbr, full in unit_abbreviations.items(): text = re.sub(rf'(?<=\d)\s*{re.escape(abbr)}\b', f' {full}', text, flags=re.IGNORECASE)
    for abbr, full in safe_abbreviations.items(): text = re.sub(r'\b' + re.escape(abbr) + r'(?!\w)', full, text, flags=re.IGNORECASE)
    problematic_abbreviations = {"N.": "North", "S.": "South", "E.": "East", "W.": "West", "p.": "page"}
    for abbr, full in problematic_abbreviations.items(): text = re.sub(r'(^|\s)' + re.escape(abbr) + r'(?!\w)', r'\1' + full, text, flags=re.IGNORECASE)
    cleaned_text = convert_numbers_to_words(text)
    cleaned_text = cleaned_text.replace('…', '.')
    cleaned_text = cleaned_text.replace('"', ' ').replace('“', ' ').replace('”', ' ')
    cleaned_text = cleaned_text.replace('()', '').replace('( )', '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# --- Python Extractors ---
def read_epub(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        book = epub.read_epub(file_path)
        text_content = []
        title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else None
        author = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else None
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text_content.append(soup.get_text(separator=' ', strip=True))
        return ' '.join(text_content), title, author

def read_pdf(file_path):
    doc = fitz.open(file_path)
    title = doc.metadata.get("title")
    author = doc.metadata.get("author")
    text_content = [page.get_text() for page in doc]
    return ' '.join(text_content), title, author

def read_mobi(file_path):
    tempdir, ext_path = mobi.extract(file_path)
    if ext_path.lower().endswith('.epub'):
        text, title, author = read_epub(ext_path)
    else:
        with open(ext_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            title, author = None, None
    shutil.rmtree(tempdir, ignore_errors=True)
    return text, title, author

def extract_text_and_metadata(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    title = os.path.splitext(os.path.basename(file_path))[0]
    author = "Unknown Author"
    raw_text = ""
    if ext == '.epub': raw_text, t, a = read_epub(file_path)
    elif ext == '.pdf': raw_text, t, a = read_pdf(file_path)
    elif ext in['.mobi', '.azw3']: raw_text, t, a = read_mobi(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: raw_text = f.read()
        t, a = None, None
    elif ext in ['.htm', '.html']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            raw_text = soup.get_text(separator=' ', strip=True)
        t, a = None, None
    else: raise ValueError(f"Unsupported file format: {ext}")
    if t: title = t
    if a: author = a
    if not raw_text.strip(): raise ValueError("No text could be extracted from the file.")
    text = clean_and_normalize_text(raw_text)
    return text, title, author

def sanitize_filename(filename):
    sanitized = re.sub(r'[\/*?:"<>|\']', "", filename)
    return sanitized.replace(" ", "_")

def ensure_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)

def show_converted_audiobooks():
    output_dir = os.path.join("Working_files", "Book")
    if not os.path.exists(output_dir): return []
    files =[os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    return files if files else[]

def basic_tts(ref_audio_input, ref_text_input, gen_file_input, speed, max_phrase_length, max_chunk_length, num_steps, progress=gr.Progress()):
    try:
        processed_audiobooks = []
        num_ebooks = len(gen_file_input)
        ebook_frac = {"extract_text": 0.05, "infer": 0.90, "mp3_meta": 0.05}

        progress(0, desc="Preprocessing reference audio for OmniVoice...")
        ref_text = ref_text_input or ""
        try:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=ref_text if ref_text.strip() else None,
            )
        except Exception as e:
            raise gr.Error(f"Error preprocessing reference audio: {e}")

        MAX_PHRASE_LENGTH = int(max_phrase_length)
        MAX_CHUNK_LENGTH_CHARS = int(max_chunk_length)
        NUM_GENERATION_STEPS = int(num_steps)

        for idx, ebook_file_data in enumerate(gen_file_input):
            current_ebook_base_progress = idx / float(num_ebooks)
            progress_offset_within_ebook = 0.0
            original_ebook_path = ebook_file_data.name
            
            if not os.path.exists(original_ebook_path):
                continue

            progress(current_ebook_base_progress, desc=f"Ebook {idx+1}/{num_ebooks}: Extracting text...")
            try:
                gen_text, ebook_title, ebook_author = extract_text_and_metadata(original_ebook_path)
            except Exception as e:
                print(f"Extraction error: {e}")
                continue

            progress_offset_within_ebook += ebook_frac["extract_text"]
            
            overall_infer_start_frac = current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks)
            temp_chunks_dir = os.path.join("Working_files", "temp_audio_chunks", sanitize_filename(ebook_title))
            ensure_directory(temp_chunks_dir)
            chunk_file_paths = []

            initial_sentences = sent_tokenize(gen_text)
            intermediate_phrases = []
            for sentence in initial_sentences:
                if len(sentence) <= MAX_PHRASE_LENGTH: intermediate_phrases.append(sentence)
                else:
                    current_part = sentence
                    while len(current_part) > MAX_PHRASE_LENGTH:
                        split_pos = -1
                        delimiters = [',', ';', '—', '–']
                        for delimiter in delimiters:
                            pos = current_part.rfind(delimiter, 0, MAX_PHRASE_LENGTH)
                            if pos > split_pos: split_pos = pos
                        if split_pos == -1: split_pos = current_part.rfind(' ', 0, MAX_PHRASE_LENGTH)
                        if split_pos == -1: split_pos = MAX_PHRASE_LENGTH
                        intermediate_phrases.append(current_part[:split_pos+1].strip())
                        current_part = current_part[split_pos+1:].strip()
                    if current_part: intermediate_phrases.append(current_part)

            text_super_chunks = []
            current_chunk = ""
            for phrase in intermediate_phrases:
                if len(current_chunk) + len(phrase) + 1 > MAX_CHUNK_LENGTH_CHARS:
                    if current_chunk: text_super_chunks.append(current_chunk)
                    current_chunk = phrase
                else:
                    if current_chunk: current_chunk += " " + phrase
                    else: current_chunk = phrase
            if current_chunk: text_super_chunks.append(current_chunk)

            del gen_text
            gc.collect()

            num_super_chunks = len(text_super_chunks)
            if num_super_chunks == 0: continue

            ebook_start_time = time.time()
            progress_updater = EbookProgressUpdater(progress, num_super_chunks, idx, num_ebooks, ebook_start_time)

            gen_config = OmniVoiceGenerationConfig(num_step=NUM_GENERATION_STEPS)
            for i, text_chunk in enumerate(text_super_chunks):
                progress_updater.set_chunk_index(i)
                chunk_progress_start = overall_infer_start_frac + (i / num_super_chunks) * (ebook_frac["infer"] / num_ebooks)
                progress_updater(chunk_progress_start)
                try:
                    with torch.no_grad():
                        audio_tensor = model.generate(text=text_chunk, voice_clone_prompt=voice_clone_prompt, speed=float(speed), generation_config=gen_config)
                    wave_chunk = audio_tensor[0].squeeze(0).cpu().numpy()
                    if wave_chunk is not None and wave_chunk.any():
                        chunk_path = os.path.join(temp_chunks_dir, f"chunk_{i:04d}.wav")
                        sf.write(chunk_path, wave_chunk, sampling_rate)
                        chunk_file_paths.append(chunk_path)
                finally:
                    if 'audio_tensor' in locals(): del audio_tensor
                    if 'wave_chunk' in locals(): del wave_chunk
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            progress_offset_within_ebook += ebook_frac["infer"]
            progress(current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks), desc=f"Ebook {idx+1}/{num_ebooks}: Finalizing MP3...")
            
            sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{idx}"
            final_mp3_dir = os.path.join("Working_files", "Book")
            final_mp3_path = os.path.join(final_mp3_dir, f"{sanitized_title}.mp3")

            try:
                concat_list_path = os.path.join(temp_chunks_dir, "concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for path in chunk_file_paths: f.write(f"file '{os.path.basename(path)}'\n")

                ensure_directory(final_mp3_dir)
                ffmpeg_command = [FFMPEG_EXE, '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt', '-c:a', 'libmp3lame', '-b:a', '192k', '-id3v2_version', '3', '-metadata', f'title={ebook_title}', '-metadata', f'artist={ebook_author}', '-metadata', f'album={ebook_title}', '-y', os.path.abspath(final_mp3_path)]
                subprocess.run(ffmpeg_command, cwd=temp_chunks_dir, check=True)
                
                # --- ADD TO LIST AND YIELD IMMEDIATELY ---
                processed_audiobooks.append(final_mp3_path)
                yield processed_audiobooks # THIS MAKES THE DOWNLOAD APPEAR NOW
                
            except Exception as e:
                print(f"FFmpeg error: {e}")
            finally:
                if os.path.exists(temp_chunks_dir): shutil.rmtree(temp_chunks_dir)

        progress(1.0, desc=f"All {num_ebooks} eBook(s) processing finished.")

    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

DEFAULT_REF_AUDIO_PATH = "default_voice.mp3"
DEFAULT_REF_TEXT = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well. The juice of lemons makes fine punch."

def create_gradio_app():
    with gr.Blocks(theme=gr.themes.Ocean()) as app:
        gr.Markdown("# eBook to Audiobook with OmniVoice")
        
        # --- INPUT SECTION ---
        # Wrapping in a Column or Group ensures they stay together vertically
        with gr.Column():
            ref_audio_input = gr.Audio(
                label="Upload Voice File (<15 sec) or Record", 
                type="filepath", 
                value=DEFAULT_REF_AUDIO_PATH
            )
            gen_file_input = gr.File(
                label="Upload eBooks (Batch Support)", 
                file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"], 
                file_count="multiple", 
                type="filepath"
            )
            generate_btn = gr.Button("Start Processing", variant="primary")

        # --- OUTPUT SECTION ---
        # Because this is NOT inside a gr.Row with the section above, 
        # it will always stay underneath.
        with gr.Column():
            audiobooks_output = gr.Files(label="Completed Audiobooks (Download as they finish)")
            show_audiobooks_btn = gr.Button("Show All Files in Output Folder", variant="secondary")

        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(label="Reference Text", lines=2, value=DEFAULT_REF_TEXT)
            speed_slider = gr.Slider(label="Speech Speed", minimum=0.3, maximum=2.0, value=1.0, step=0.1)
            num_steps_slider = gr.Slider(label="Generation Steps", minimum=10, maximum=100, value=70, step=1)
            max_phrase_slider = gr.Slider(label="Max Phrase Length", minimum=200, maximum=2000, value=300, step=50)
            max_chunk_slider = gr.Slider(label="Max Chunk Length", minimum=500, maximum=4000, value=800, step=50)

        generate_btn.click(
            basic_tts,
            inputs=[ref_audio_input, ref_text_input, gen_file_input, speed_slider, max_phrase_slider, max_chunk_slider, num_steps_slider],
            outputs=[audiobooks_output]
        )
        show_audiobooks_btn.click(show_converted_audiobooks, inputs=[], outputs=[audiobooks_output])
        
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.queue(default_concurrency_limit=2).launch(server_name=os.environ.get("OMNIVOICE_HOST", "127.0.0.1"), server_port=int(os.environ.get("OMNIVOICE_PORT", 7860)), max_threads=10)
