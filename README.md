# Ebook to Audiobook with OmniVoice

Turn your eBooks (epub, pdf, mobi, txt, html) into high-quality audiobooks using the OmniVoice text-to-speech model. 

Built for [Pinokio](https://pinokio.computer/) - the easiest way to install and run AI applications locally.

<img width="1464" height="1625" alt="image" src="https://github.com/user-attachments/assets/3f051597-c48d-4e4e-8745-6d4c67bb58fa" />

## Features
* **Zero Calibre Dependency:** Natively extracts text from EPUB, PDF, MOBI, and more.
* **Voice Cloning:** Upload a 15-second voice sample to narrate your entire book.
* **Smart Chunking:** Built to avoid Out-Of-Memory (OOM) errors on low VRAM machines.
* **Automatic Metadata:** MP3 files are automatically tagged with the book's title and author.
* **Multi-Platform:** Tested and works on Windows (NVIDIA/AMD/CPU) and macOS (Apple Silicon).

## How to Install
1. Download and install [Pinokio](https://pinokio.computer/).
2. Click the link below (or copy/paste it into your browser):
   [pinokio://download?uri=https://github.com/quantumlump/Ebook-to-Audiobook-with-OmniVoice](pinokio://download?uri=https://github.com/quantumlump/Ebook-to-Audiobook-with-OmniVoice)
3. Click "Download", then click "Install".

## Usage
Once installed, click **Start** to open the Gradio Web UI. Use the provided reference voice or upload your own, upload one or more eBooks, and click "Start Processing". Your completed audiobooks will appear in the UI to download, and are also saved in the `Working_files/Book` folder.
