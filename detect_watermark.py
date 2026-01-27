
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator
import os

def detect_watermark(image_path, text_labels):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = Accelerator().device

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    draw = ImageDraw.Draw(image)
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        draw.rectangle(box, outline="red", width=3)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    image.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    image_path = "input/001.jpg"
    text_labels = [["a watermark"]]
    detect_watermark(image_path, text_labels)
