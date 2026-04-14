module.exports = {
  run:[
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        path: ".",
        message: ["uv pip install wheel"],
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "uv pip install -r app/requirements.txt",
      },
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
        },
      },
    },
    // NEW STEP: Pre-download model weights and NLTK data during installation
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          // This forces the model to save INSIDE the app folder rather than your main C: drive
          HF_HOME: "hf_cache" 
        },
        message: "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice', load_asr=True)\""
      }
    }
  ],
}
