module.exports = {
  daemon: true,
  run:[
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          PYTHONUNBUFFERED: "1",
          OMNIVOICE_PORT: "{{port}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          // NEW VARIABLES: Force strict offline mode
          HF_HUB_OFFLINE: "1",
          TRANSFORMERS_OFFLINE: "1",
          // Point to the local folder we downloaded the weights to in install.js
          HF_HOME: "hf_cache",
          // Stop Gradio from trying to ping its analytics servers
          GRADIO_ANALYTICS_ENABLED: "False"
        },
        message: ["python -u app/app.py"],
        on:[
          {
            event: "/(http:\\/\\/[0-9.:]+)/",
            done: true,
          },
        ],
      },
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
      },
    },
  ],
}
