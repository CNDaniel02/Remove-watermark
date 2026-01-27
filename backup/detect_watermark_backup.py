
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator
import os

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression to remove overlapping boxes
    boxes: tensor of shape (N, 4) in format [x1, y1, x2, y2]
    scores: tensor of shape (N,)
    iou_threshold: IoU threshold for suppression
    """
    if len(boxes) == 0:
        return []
    
    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort by scores in descending order
    _, indices = torch.sort(scores, descending=True)
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
        
        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union area
        remaining_areas = areas[indices[1:]]
        union = areas[current] + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        mask = iou < iou_threshold
        indices = indices[1:][mask]
    
    return keep

class WatermarkDetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=None):
        if device is None:
            self.device = Accelerator().device
        else:
            self.device = device
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, image_path, text_labels, box_threshold=0.1, text_threshold=0.1, output_dir="output", padding=10):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        img_area = img_width * img_height

        # Collect all valid boxes and scores after filtering
        valid_boxes = []
        valid_scores = []
        valid_labels = []
        
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            if score < box_threshold:
                continue
            box_list = box.tolist()
            
            # Calculate dimensions
            w = box_list[2] - box_list[0]
            h = box_list[3] - box_list[1]
            
            if h <= 0 or w <= 0:
                continue

            # Calculate metrics
            aspect_ratio = w / h
            area_ratio = (w * h) / img_area
            width_ratio = w / img_width
            height_ratio = h / img_height
            
            # Apply filters
            # Aspect Ratio: 2.0 <= (w/h) <= 4.0
            if not (2.0 <= aspect_ratio <= 4.0):
                continue
            # Area Ratio: 0.002 <= (w*h)/(W*H) <= 0.08
            if not (0.002 <= area_ratio <= 0.08):
                continue
            # Width Ratio: 0.08 <= w/W <= 0.45
            if not (0.08 <= width_ratio <= 0.45):
                continue
            # Height Ratio: 0.02 <= h/H <= 0.20
            if not (0.02 <= height_ratio <= 0.20):
                continue
            
            # This box passed all filters, add to valid list
            valid_boxes.append(box)
            valid_scores.append(score)
            valid_labels.append(label)
        
        # Apply NMS to remove overlapping boxes
        if len(valid_boxes) > 0:
            boxes_tensor = torch.stack(valid_boxes)
            scores_tensor = torch.stack(valid_scores)
            
            # Apply NMS with IoU threshold of 0.5
            keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            
            # Draw only the boxes that survived NMS
            for idx in keep_indices:
                box = valid_boxes[idx]
                score = valid_scores[idx]
                label = valid_labels[idx]
                
                box_list = box.tolist()
                
                # Add padding and clamp to image boundaries
                x1, y1, x2, y2 = box_list
                padded_box = [
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(img_width, x2 + padding),
                    min(img_height, y2 + padding)
                ]

                padded_box_rounded = [round(x, 2) for x in padded_box]
                original_box_rounded = [round(x, 2) for x in box_list]
                print(f"Detected {label} with confidence {round(score.item(), 3)} at location {original_box_rounded}")
                draw.rectangle(padded_box_rounded, outline="red", width=3)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        image.save(output_path)
        print(f"Saved result to {output_path}")

if __name__ == "__main__":
    detector = WatermarkDetector(model_id="IDEA-Research/grounding-dino-tiny")
    text_labels = [["a transparent watermark", "a faint logo", "semi-transparent text"]]
    
    # Process all images in input folder
    input_dir = "input"
    output_dir = "output"
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        exit(1)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process only first 10 images for testing
    test_limit = 10
    image_files = image_files[:test_limit]
    print(f"Processing first {len(image_files)} images for testing...")
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {i}/{len(image_files)}: {image_file}")
        detector.detect(image_path, text_labels)
        print(f"Completed {image_file}")
        print("-" * 50)
    
    print(f"Batch processing completed! Results saved in {output_dir}")
