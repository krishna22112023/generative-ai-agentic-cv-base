from glob import glob
import os

def convert_annotations(
    annotations,
    class_map,
    image_size,
    format):
    """
    Convert annotations to YOLO or COCO format and save class names to a txt file.
    
    :param annotations: List of dicts with 'box_2d' and 'label'
    :param class_map: classes mapped to id)
    :param format: 'yolo' or 'coco'
    :param image_size: Tuple (width, height)
    :return: Converted annotations
    """
    (width, height) = image_size
    
    if format == "yolo":
        yolo_lines = []
        for ann in annotations:
            box = ann["box_2d"]
            label = ann["label"]
            class_id = class_map[label]
            
            x_min, y_min, x_max, y_max = box
            center_x = (x_min + x_max) / 2.0 / width
            center_y = (y_min + y_max) / 2.0 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        return yolo_lines

def get_processed_files(path:str):
    #Extract the last file of every image subfolder since it is the image processed in the last step.
    subfolders = glob(os.path.join(path, "*", ""))
    last_jpeg_files = []
    for subfolder in subfolders:
        jpeg_files = glob(os.path.join(subfolder, "*.jpg"))
        jpeg_files.sort(key=os.path.getmtime)
        last_jpeg_files.append(jpeg_files[-1])

    return last_jpeg_files


if __name__ == "__main__":
    print(get_processed_files("/Users/krishnaiyer/generative-ai-agentic-cv-base/data/processed/DAWN/Rain"))