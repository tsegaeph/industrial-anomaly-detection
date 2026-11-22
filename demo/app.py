"""Gradio demo: upload a product image, choose its category, get verdict, score and heatmap."""
import gradio as gr

from src.inference.predictor import AnomalyPredictor, render_overlay

predictor = AnomalyPredictor("models", "thresholds.json", rule="p99")

INTRO = """# Industrial Visual Anomaly Detection
PatchCore trained **only on normal images** of each product (MVTec AD). Pick the product category,
upload an image, and the model scores how much it deviates from normal and highlights where.
Works best on images taken like the MVTec AD photos. Noise, blur and lighting changes degrade it."""


def inspect(image, category, scale):
    if image is None:
        return None, None, "Please upload an image."
    pred = predictor.predict(image, category, scale)
    verdict = "DEFECTIVE" if pred.is_anomalous else "NORMAL"
    summary = (f"### Verdict: {verdict}\n\nanomaly score **{pred.score:.3f}**, "
               f"decision threshold **{pred.threshold:.3f}**")
    return pred.image, render_overlay(pred), summary


with gr.Blocks(title="Industrial Anomaly Detection") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Product image")
            category = gr.Dropdown(predictor.categories, value=predictor.categories[0], label="Product category")
            scale = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Threshold scale (lower = more sensitive)")
            run = gr.Button("Inspect", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            with gr.Row():
                out_img = gr.Image(label="Input (resized)")
                out_map = gr.Image(label="Anomaly heatmap")
    run.click(inspect, [image, category, scale], [out_img, out_map, summary])

demo.launch()