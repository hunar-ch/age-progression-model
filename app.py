"""
Gradio GUI for age progression/regression.

Two views:
  - Single step: pick one of the 6 steps, see the result, download it.
  - All steps: run once, see all 6 outputs side by side.

Both views reuse the same cached model_utils.generate(), so switching
between steps for an already-uploaded image never re-runs the network.
"""

import gradio as gr

from model_utils import generate, generate_all_steps, get_arrays, ALPHA_MAP
import metrics_utils

STEP_ORDER = ["-30", "-20", "-10", "+10", "+20", "+30"]


def on_step_change(image, step_label, show_metrics):
    """Single-step view: regenerate (or pull from cache) when image, step, or the
    metrics toggle changes. Metrics are only computed when the checkbox is on."""
    if image is None:
        return None, ""
    alpha = ALPHA_MAP[step_label]
    result = generate(image, alpha)

    if not show_metrics:
        return result, ""

    orig_np, current_np = get_arrays(image, alpha)
    metrics = metrics_utils.compute_metrics(orig_np, current_np, alpha)
    return result, metrics_utils.format_metrics_markdown(metrics)


def on_generate_all(image):
    """Run all 6 steps, return each image plus its own metrics markdown, in step order."""
    if image is None:
        return [None] * len(STEP_ORDER) + [""] * len(STEP_ORDER)

    results = generate_all_steps(image)

    images = [results[label] for label in STEP_ORDER]
    metrics_md = []
    for label in STEP_ORDER:
        alpha = ALPHA_MAP[label]
        orig_np, current_np = get_arrays(image, alpha)
        m = metrics_utils.compute_metrics(orig_np, current_np, alpha)
        psnr_str = f"{m['psnr']:.2f} dB" if m["psnr"] != float("inf") else "Baseline"
        metrics_md.append(
            f"**PSNR:** {psnr_str}  \n**SSIM:** {m['ssim']:.4f}  \n**Identity sim:** {m['identity_similarity']:.4f}"
        )

    return images + metrics_md


CUSTOM_CSS = """
.gradio-image img {
    padding: 0 !important;
}
.metrics-table table {
    margin-left: auto;
    margin-right: auto;
    background: transparent !important;
}
.metrics-table table th,
.metrics-table table td {
    background: transparent !important;
}
"""

with gr.Blocks(title="Face Age Progression", css=CUSTOM_CSS) as demo:
    gr.Markdown("## Age Progression / Regression")
    gr.Markdown(
        "Upload a face photo, then pick an age shift. "
    )

    with gr.Tab("Single step"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Upload a face photo**")
                inp = gr.Image(type="pil", label="Upload a face photo", show_label=False, height=512, container=False)
            with gr.Column(scale=1):
                gr.Markdown("**Result**")
                out = gr.Image(label="Result", show_label=False, height=512, container=False)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="panel-box"):
                    step = gr.Radio(
                        choices=STEP_ORDER,
                        value="+20",
                        label="Age shift (years)",
                    )
                    show_metrics = gr.Checkbox(
                        label="Show metrics (PSNR / SSIM / FaceNet identity similarity)",
                        value=False,
                    )
            with gr.Column(scale=1):
                metrics_out = gr.Markdown(elem_classes="metrics-table")

        # Live update: re-render whenever the image, selected step, or metrics toggle changes.
        inp.change(fn=on_step_change, inputs=[inp, step, show_metrics], outputs=[out, metrics_out])
        step.change(fn=on_step_change, inputs=[inp, step, show_metrics], outputs=[out, metrics_out])
        show_metrics.change(fn=on_step_change, inputs=[inp, step, show_metrics], outputs=[out, metrics_out])

    with gr.Tab("All steps"):
        with gr.Row():
            gallery_inp = gr.Image(type="pil", label="Upload a face photo")
        gallery_btn = gr.Button("Generate all 6 steps", variant="primary")

        step_images = []
        step_metrics = []
        with gr.Row():
            for label in STEP_ORDER:
                with gr.Column(min_width=140):
                    img = gr.Image(label=f"{label} yrs", show_label=True, height=180)
                    md = gr.Markdown()
                    step_images.append(img)
                    step_metrics.append(md)

        gallery_btn.click(fn=on_generate_all, inputs=gallery_inp, outputs=step_images + step_metrics)


if __name__ == "__main__":
    demo.launch()