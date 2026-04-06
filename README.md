# Ebook to Audiobook with OmniVoice

Turn your eBooks (epub, pdf, mobi, txt, html) into high-quality audiobooks using the OmniVoice text-to-speech model. 

Built for [Pinokio](https://pinokio.computer/) - the easiest way to install and run AI applications locally.

## Features
* **Zero Calibre Dependency:** Natively extracts text from EPUB, PDF, MOBI, and more.
* **Voice Cloning:** Upload a 15-second voice sample to narrate your entire book.
* **Smart Chunking:** Built to avoid Out-Of-Memory (OOM) errors on low VRAM machines.
* **Automatic Metadata:** MP3 files are automatically tagged with the book's title and author.
* **Multi-Platform:** Tested and works on Windows (NVIDIA/AMD/CPU) and macOS (Apple Silicon).

## How to Install
1. Download and install[Pinokio](https://pinokio.computer/).
2. Open Pinokio and paste the following URL into the search/address bar at the top:
   `https://github.com/quantumlump/Ebook-to-Audiobook-with-OmniVoice/upload`
3. Click "Download", then click "Install".

## Usage
Once installed, click **Start** to open the Gradio Web UI. Upload a reference voice, upload one or more eBooks, and click "Start Processing". Your completed audiobooks will appear in the UI to download, and are also saved in the `Working_files/Book` folder.