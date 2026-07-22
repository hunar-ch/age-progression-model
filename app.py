"""
Gradio GUI for age progression/regression.

Two views:
  - Single step: pick one of the 6 steps, see the result, download it.
  - All steps: run once, see all 6 outputs side by side.

Both views reuse the same cached model_utils.generate(), so switching
between steps for an already-uploaded image never re-runs the network.
"""

import gradio as gr

from model_utils import generate, generate_all_steps, ALPHA_MAP

STEP_ORDER = ["-30", "-20", "-10", "+10", "+20", "+30"]


def on_step_change(image, step_label):
    """Single-step view: regenerate (or pull from cache) when image or step changes."""
    if image is None:
        return None
    alpha = ALPHA_MAP[step_label]
    return generate(image, alpha)


def on_generate_all(image):
    """Gallery view: run all 6 steps, return as a list of (image, caption) for gr.Gallery."""
    if image is None:
        return []
    results = generate_all_steps(image)
    return [(results[label], f"{label} yrs") for label in STEP_ORDER]


with gr.Blocks(title="Face Age Progression") as demo:
    gr.Markdown("## Age Progression / Regression")
    gr.Markdown(
        "Upload a face photo, then pick an age shift. "
        "Steps map to alpha: -30/-20/-10 → -1.5/-1.0/-0.5, +10/+20/+30 → +0.5/+1.0/+1.5."
    )

    with gr.Tab("Single step"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload a face photo", height=512, width=512)
                step = gr.Radio(
                    choices=STEP_ORDER,
                    value="+20",
                    label="Age shift (years)",
                )
            with gr.Column():
                out = gr.Image(label="Result", height=512, width=512)

        # Live update: re-render whenever the image or the selected step changes.
        inp.change(fn=on_step_change, inputs=[inp, step], outputs=out)
        step.change(fn=on_step_change, inputs=[inp, step], outputs=out)

    with gr.Tab("All steps"):
        with gr.Row():
            gallery_inp = gr.Image(type="pil", label="Upload a face photo")
        gallery_btn = gr.Button("Generate all 6 steps", variant="primary")
        gallery_out = gr.Gallery(label="Age progression / regression", columns=6, height="auto")

        gallery_btn.click(fn=on_generate_all, inputs=gallery_inp, outputs=gallery_out)


if __name__ == "__main__":
    demo.launch()