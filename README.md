# Face Age Progression / Regression GUI

A Gradio-based GUI for a CycleGAN face-aging model. Upload a photo, pick an age
shift from **-30 to +30 years**, and see the result — either one step at a
time or all six side by side, with optional PSNR / SSIM / FaceNet identity
similarity metrics.

Built on top of the generator architecture from
[HasnainRaz/Fast-AgingGAN](https://github.com/HasnainRaz/Fast-AgingGAN) (MIT
licensed).

## Features

- **Single step tab** — pick one of 6 age-shift steps, see the result update
  live, with an optional metrics panel (PSNR / SSIM / FaceNet identity
  similarity).
- **All steps tab** — generate all 6 steps at once, each with its own metrics
  underneath.
- Runs a single forward pass per uploaded image — switching between the 6
  steps just rescales a cached "aging residual," so it feels instant after
  the first generation.
- Uses Apple Silicon (MPS) GPU acceleration automatically if available,
  otherwise falls back to CPU.

## How the age-shift steps work

The model outputs one "fully aged" prediction. Each of the 6 steps scales
that change by a different **alpha**:

| Step | Alpha |
|---|---|
| -30 yrs | -1.5 |
| -20 yrs | -1.0 |
| -10 yrs | -0.5 |
| +10 yrs | +0.5 |
| +20 yrs | +1.0 |
| +30 yrs | +1.5 |

## Project structure

```
aging_gui/
├── app.py              # Gradio app (entrypoint)
├── model_utils.py       # Model loading + cached inference logic
├── metrics_utils.py      # PSNR / SSIM / FaceNet identity similarity
├── models.py             # Generator/Discriminator architecture (Fast-AgingGAN)
├── test_model.py          # Standalone sanity check (run before the GUI)
├── requirements.txt
├── pretrained_model/
│   └── state_dict.pth   # Model checkpoint (included, ~11MB)
└── README.md
```

## Setup

### 1. Clone the repo

The checkpoint is included, so this is a self-contained clone — no separate
downloads needed.

```bash
git clone https://github.com/hunar-ch/age-progression-model
cd age-progression-model
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model checkpoint

The trained generator checkpoint is already included at:

```
pretrained_model/state_dict.pth
```

No download needed — it's committed directly in this repo (~11MB, well
under GitHub's file size limits). It matches the architecture in
`models.py`: `Generator(ngf=32, n_residual_blocks=9)`. If you swap in a
checkpoint trained with different hyperparameters, update the `load_model()`
call in `model_utils.py` to match.

### 4. (Recommended) Run the sanity check first

Before launching the GUI, confirm the model loads correctly and that the age
direction actually looks right (younger on the left, older on the right):

```bash
python test_model.py /path/to/a/face/photo.jpg
```

This saves a comparison grid to `test_output.png` and also displays it.

### 5. Launch the GUI

```bash
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`) — open it
in your browser.

## Notes

- **Device**: automatically uses `mps` on Apple Silicon if available,
  otherwise `cpu`. There's currently no CUDA-specific handling — if you're on
  an NVIDIA GPU, `torch.device("cuda")` would need to be added to
  `get_device()` in `model_utils.py`.
- **Metrics panel**: `facenet-pytorch` is only loaded lazily, the first time
  you check the metrics checkbox — so plain image generation stays fast even
  if that dependency isn't needed for a given session.
- **Non-square uploads**: the model always processes images at a fixed
  512×512 resolution, so the "Result" panel is always square regardless of
  your upload's original aspect ratio.

## License

The `models.py` architecture is reused from
[HasnainRaz/Fast-AgingGAN](https://github.com/HasnainRaz/Fast-AgingGAN)
(MIT License).