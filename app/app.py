import os
import shutil
import tempfile
import time
import io
import re
import gc
import subprocess
import warnings
import starlette.datastructures
import traceback

# --- 1. THE PINOKIO WATCHER CRASH FIX ---
# Pinokio watches the current working directory for file changes. If we stream
# a large upload to a local temp folder, it triggers hundreds of rapid file events,
# instantly overloading and crashing Pinokio on Windows.
# FIX: Move Gradio's temp folder OUTSIDE the Pinokio workspace to the Windows system temp folder.
SYSTEM_TEMP = tempfile.gettempdir()
LOCAL_TEMP = os.path.join(SYSTEM_TEMP, "omnivoice_gradio_temp")

if os.path.exists(LOCAL_TEMP):
    try:
        shutil.rmtree(LOCAL_TEMP, ignore_errors=True)
    except Exception:
        pass
os.makedirs(LOCAL_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = LOCAL_TEMP

# --- 2. THE FASTAPI RAM SPOOL FIX ---
# Prevent FastAPI from spooling uploads < 100MB to disk at all.
# This keeps the eBook entirely in RAM during upload, completely bypassing 
# Windows disk locks and file-watcher events, making the upload instant!
starlette.datastructures.UploadFile.spool_max_size = 100 * 1024 * 1024 


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
from ebooklib import epub, ITEM_DOCUMENT, ITEM_COVER, ITEM_IMAGE
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 2. Specifically force Transformers and Gradio to stop being quiet
import transformers
transformers.utils.logging.set_verbosity_info()

# 3. If you want to see every single web request Gradio/FastAPI receives:
logging.getLogger("torch").setLevel(logging.DEBUG)
transformers.utils.logging.set_verbosity_debug()


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
    text = re.sub(r'\[\d+(?:,\s*\d+|-\d+)*\]', '', text)
    text = re.sub(r'\[(?:sic|ibid|ref|page\s\d+)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=\w)[\*\†\‡\§]\d*', '', text)
    text = re.sub(r'^\s*[\*\†\‡\§]\d*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=[a-zA-Z])\.\d{1,3}\b', '.', text)
    text = re.sub(r'\(\s*[A-Z][a-z]+,?\s+\d{4}\s*\)', '', text)
    text = re.sub(r'^\s*\d+\b\s*', '', text, flags=re.MULTILINE)
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

# --- Extractors (Updated for Covers & Metadata) ---
def get_epub_meta(book, namespace, name):
    meta = book.get_metadata(namespace, name)
    if meta and len(meta) > 0 and isinstance(meta[0], tuple) and len(meta[0]) > 0:
        return meta[0][0]
    return None

def read_epub(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        book = epub.read_epub(file_path)
        text_content =[]
        
        # Robust metadata extraction
        title = get_epub_meta(book, 'DC', 'title')
        author = get_epub_meta(book, 'DC', 'creator')
        
        # Cover extraction
        cover_bytes = None
        try:
            for item in book.get_items_of_type(ITEM_COVER):
                cover_bytes = item.get_content()
                break
            
            # Fallback cover extraction (if cover is tagged as image)
            if not cover_bytes:
                for item in book.get_items_of_type(ITEM_IMAGE):
                    name = item.get_name().lower()
                    item_id = getattr(item, 'id', '').lower()
                    if 'cover' in name or 'cover' in item_id:
                        cover_bytes = item.get_content()
                        break
        except Exception as e:
            print(f"EPUB cover extraction warning: {e}")
            
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for unwanted in soup(["aside", "footnote", "reftag", "nav", "footer"]):
                unwanted.decompose()
            text_content.append(soup.get_text(separator=' ', strip=True))
            
        return ' '.join(text_content), title, author, cover_bytes

def read_pdf(file_path):
    doc = fitz.open(file_path)
    title = doc.metadata.get("title")
    author = doc.metadata.get("author")
    
    # Extract First Page as Cover Image
    cover_bytes = None
    try:
        if len(doc) > 0:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            cover_bytes = pix.tobytes("jpeg")
    except Exception as e:
        print(f"PDF cover extraction warning: {e}")
        
    text_content = [page.get_text() for page in doc]
    return ' '.join(text_content), title, author, cover_bytes

def read_mobi(file_path):
    tempdir, ext_path = mobi.extract(file_path)
    cover_bytes = None
    if ext_path.lower().endswith('.epub'):
        text, title, author, cover_bytes = read_epub(ext_path)
    else:
        with open(ext_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            title, author = None, None
    shutil.rmtree(tempdir, ignore_errors=True)
    return text, title, author, cover_bytes

def extract_text_and_metadata(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    raw_text = ""
    cover_bytes = None
    t, a = None, None
    
    if ext == '.epub': raw_text, t, a, cover_bytes = read_epub(file_path)
    elif ext == '.pdf': raw_text, t, a, cover_bytes = read_pdf(file_path)
    elif ext in['.mobi', '.azw3']: raw_text, t, a, cover_bytes = read_mobi(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: raw_text = f.read()
    elif ext in['.htm', '.html']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            raw_text = soup.get_text(separator=' ', strip=True)
    else: raise ValueError(f"Unsupported file format: {ext}")
    
    # Fallback Metadata Fix (if the book properties are left blank/untitled)
    if t and t.strip() and t.strip().lower() != "untitled":
        title = t.strip()
    else:
        title = base_name.replace("_", " ").title()
        
    if a and a.strip() and a.strip().lower() != "untitled":
        author = a.strip()
    else:
        author = "Unknown Author"
        
    if not raw_text.strip(): raise ValueError("No text could be extracted from the file.")
    text = clean_and_normalize_text(raw_text)
    
    return text, title, author, cover_bytes

def sanitize_filename(filename):
    if not filename:
        return "unknown_file"
    # Remove illegal Windows characters
    sanitized = re.sub(r'[\/*?:"<>|\']', "", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # IMPORTANT: Windows cannot handle folder names ending in a dot or space
    sanitized = sanitized.strip(". ")
    # Limit length to 100 chars to avoid "Path too long" errors
    return sanitized[:100]

def ensure_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)

def show_converted_audiobooks():
    # Use absolute paths so Gradio can reliably serve the downloads
    output_dir = os.path.abspath(os.path.join("Working_files", "Book"))
    if not os.path.exists(output_dir): return []
    files =[os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    return files if files else[]

def basic_tts(ref_audio_input, ref_text_input, gen_file_input, speed, max_phrase_length, max_chunk_length, num_steps, cfg, progress=gr.Progress()):
    try:
        processed_audiobooks =[]
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
                # Includes cover bytes and robust metadata mapping
                gen_text, ebook_title, ebook_author, cover_bytes = extract_text_and_metadata(original_ebook_path)
            except Exception as e:
                print(f"Extraction error: {e}")
                continue

            progress_offset_within_ebook += ebook_frac["extract_text"]
            
            overall_infer_start_frac = current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks)
            temp_chunks_dir = os.path.abspath(os.path.join("Working_files", "temp_audio_chunks", sanitize_filename(ebook_title)))
            ensure_directory(temp_chunks_dir)
            chunk_file_paths =[]

            initial_sentences = sent_tokenize(gen_text)
            intermediate_phrases =[]
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

            text_super_chunks =[]
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

            gen_config = OmniVoiceGenerationConfig(num_step=NUM_GENERATION_STEPS, guidance_scale=float(cfg))
            for i, text_chunk in enumerate(text_super_chunks):
                progress_updater.set_chunk_index(i)
                print(f"\n[DEBUG] Starting Chunk {i+1}/{num_super_chunks}")
                print(f"[DEBUG] Text Snippet: {text_chunk[:100]}...")
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
            final_mp3_dir = os.path.abspath(os.path.join("Working_files", "Book"))
            final_mp3_path = os.path.join(final_mp3_dir, f"{sanitized_title}.mp3")

            try:
                concat_list_path = os.path.join(temp_chunks_dir, "concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for path in chunk_file_paths: f.write(f"file '{os.path.basename(path)}'\n")

                ensure_directory(final_mp3_dir)
                
                # Base FFmpeg command
                ffmpeg_command =[FFMPEG_EXE, '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt']
                
                # Handling Cover Image Output map for MP3 tagging
                has_cover = False
                cover_path = os.path.join(temp_chunks_dir, "cover.jpg")
                if cover_bytes:
                    try:
                        with open(cover_path, "wb") as f:
                            f.write(cover_bytes)
                        has_cover = True
                    except Exception as e:
                        print(f"Failed to write cover image: {e}")
                
                if has_cover:
                    # Map the audio array [0:a] and the image [1:v] so it embeds the image
                    ffmpeg_command.extend([
                        '-i', 'cover.jpg',
                        '-map', '0:a',
                        '-map', '1:v',
                        '-c:v', 'mjpeg',
                        '-disposition:v', 'attached_pic'
                    ])
                
                # Add Author/Title/Album Metadata explicitly 
                ffmpeg_command.extend([
                    '-c:a', 'libmp3lame', 
                    '-b:a', '192k', 
                    '-id3v2_version', '3', 
                    '-metadata', f'title={ebook_title}', 
                    '-metadata', f'artist={ebook_author}',   # Audiobooks use "artist" for Authors
                    '-metadata', f'album={ebook_title}',     # Treat the entire book as an Album
                    '-y', os.path.abspath(final_mp3_path)
                ])

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
        print("\n" + "="*50)
        print("CRITICAL ERROR DURING TTS PROCESSING")
        traceback.print_exc()
        print("="*50 + "\n")
        raise gr.Error(f"An error occurred: {str(e)}")

DEFAULT_REF_AUDIO_PATH = "default_voice.mp3"
DEFAULT_REF_TEXT = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well. The juice of lemons makes fine punch."

def create_gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("# eBook to Audiobook with OmniVoice")
        
        # --- INPUT SECTION ---
        with gr.Column():
            ref_audio_input = gr.Audio(
                label="Use Default Voice, Upload Voice File, or Use Record Button", 
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
        with gr.Column():
            gr.Markdown("### Current Batch Progress")
            batch_output = gr.Files(label="Audiobooks Finishing in Current Batch")
            
            gr.Markdown("### Completed Audiobooks Library")
            library_output = gr.Files(label="All Saved Audiobooks in Output Folder")
            show_audiobooks_btn = gr.Button("Refresh / Show All Saved Audiobooks", variant="secondary")

        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(label="Reference Text", lines=2, value=DEFAULT_REF_TEXT)
            speed_slider = gr.Slider(label="Speech Speed", minimum=0.3, maximum=2.0, value=1.0, step=0.1)
            num_steps_slider = gr.Slider(label="Generation Steps", minimum=10, maximum=100, value=70, step=1)
            cfg_slider = gr.Slider(label="CFG Scale (Guidance)", minimum=1.0, maximum=7.0, value=4.0, step=0.1, info="2.0 is recommended for naturalness; 4.0 is punchier.")
            max_phrase_slider = gr.Slider(label="Max Phrase Length", minimum=200, maximum=2000, value=300, step=50)
            max_chunk_slider = gr.Slider(label="Max Chunk Length", minimum=500, maximum=4000, value=800, step=50)

        # 1. Target the batch progress to the batch_output component
        generate_btn.click(
            basic_tts,
            inputs=[ref_audio_input, ref_text_input, gen_file_input, speed_slider, max_phrase_slider, max_chunk_slider, num_steps_slider, cfg_slider],
            outputs=[batch_output]
        )
        
        # 2. Target the refresh button to the entirely separate library_output component
        show_audiobooks_btn.click(
            show_converted_audiobooks, 
            inputs=[], 
            outputs=[library_output],
            queue=False  # <--- ADD THIS TO BYPASS THE QUEUE
        )
        
        # Optional Bonus: Automatically load previously completed books when the web page is opened!
        app.load(show_converted_audiobooks, inputs=[], outputs=[library_output], queue=False)
        
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.queue(default_concurrency_limit=2).launch(theme=gr.themes.Ocean(), server_name=os.environ.get("OMNIVOICE_HOST", "127.0.0.1"), server_port=int(os.environ.get("OMNIVOICE_PORT", 7860)), max_threads=10, show_error=True, quiet=False)
