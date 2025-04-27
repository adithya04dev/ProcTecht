# import os
# from ultralytics import YOLO
# import cv2
# import torch

# import time 

# def process_images():
#     # Create results directory if it doesn't exist
#     print("CUDA Version:", torch.version)
#     print("GPU Available:", torch.cuda.device_count() > 0)
#     results_dir = os.path.join('trained_models', 'results')
#     os.makedirs(results_dir, exist_ok=True)

#     # Dictionary mapping image folders to their corresponding model files
#     model_mapping = {
#         # 'climber': 'climber.pt',
#         # 'fights': 'fights.pt',
#         # 'fire_smoke': 'fire_smoke.pt',
#         # 'guns_knives': 'guns_knives.pt',
#         'realtime_accidents': 'realtime_accident.pt'
#     }

#     # Process each category
#     for category, model_file in model_mapping.items():
#         print(f"Processing {category} images...")

#         # Create category-specific results directory
#         category_results_dir = os.path.join(results_dir, category)
#         os.makedirs(category_results_dir, exist_ok=True)

#         # Load the corresponding model
#         model_path = os.path.join('trained_models', 'model_weights', model_file)
#         if not os.path.exists(model_path):
#             print(f"Warning: Model file {model_file} not found!")
#             print(model_path)
#             continue

#         model = YOLO(model_path).to('cuda')

#         # Process all images in the category folder
#         image_dir = os.path.join('trained_models', 'images', category)
#         if not os.path.exists(image_dir):
#             print(f"Warning: Image directory {category} not found!")
#             print(image_dir)
#             continue

#         # Supported image extensions
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

#         # Process each image in the category folder
#         for image_file in os.listdir(image_dir):
#             if os.path.splitext(image_file.lower())[1] in image_extensions:
#                 image_path = os.path.join(image_dir, image_file)
#                 try:
#                     # Run prediction
#                     results = model(image_path)

#                     # Get the result image with boxes drawn
#                     result_image = results[0].plot()

#                     # Save the result
#                     output_path = os.path.join(category_results_dir, image_file)
#                     cv2.imwrite(output_path, result_image)

#                     print(f"Processed: {image_file}")

#                 except Exception as e:
#                     print(f"Error processing {image_file}: {str(e)}")

# if __name__ == "__main__":
#     ti=time.time()
#     process_images()

#     print("Processing complete!")
#     print(f"Time Taken: {(time.time()-ti)}")  


    #batch inference



import os
from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import numpy as np
from typing import List, Tuple
import time

def load_images(image_paths: List[str], batch_size: int = 10) -> List[List[str]]:
    """Split image paths into batches."""
    return [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

def process_images(batch_size: int = 10):
    print("CUDA Version:", torch.__version__)
    print("GPU Available:", torch.cuda.device_count() > 0)
    
    # Create results directory
    results_dir = Path('trained_models/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model mapping
    model_mapping = {
        # 'realtime_accidents': 'realtime_accident.pt',
        # Uncomment to add more categories
        # 'climber': 'climber.pt',
        # 'fights': 'fights.pt',
        'fire_smoke': 'fire_smoke.pt',
        # 'guns_knives': 'guns_knives.pt',
    }
    
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Process each category
    for category, model_file in model_mapping.items():
        print(f"\nProcessing {category} images...")
        
        # Setup paths
        category_results_dir = results_dir / category
        category_results_dir.mkdir(exist_ok=True)
        
        model_path = Path('trained_models/model_weights') / model_file
        if not model_path.exists():
            print(f"Warning: Model file {model_file} not found at {model_path}")
            continue
            
        # Load model
        model = YOLO(str(model_path)).to('cuda')
        
        # Get image paths
        image_dir = Path('trained_models/images') / category
        if not image_dir.exists():
            print(f"Warning: Image directory not found at {image_dir}")
            continue
            
        # Collect all valid image paths
        image_paths = [
            str(f) for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No valid images found in {image_dir}")
            continue
            
        # Process images in batches
        batches = load_images(image_paths, batch_size)
        total_batches = len(batches)
        
        for batch_idx, batch in enumerate(batches, 1):
            try:
                # Run batch prediction
                results = model(batch, stream=True)  # stream=True for better memory efficiency
                
                # Process each result in the batch
                for img_path, result in zip(batch, results):
                    result_image = result.plot()
                    output_path = category_results_dir / Path(img_path).name
                    cv2.imwrite(str(output_path), result_image)
                
                print(f"Processed batch {batch_idx}/{total_batches} "
                      f"({len(batch)} images)")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue

if __name__ == "__main__":
    start_time = time.time()
    process_images(batch_size=16)  # Adjust batch size as needed
    
    duration = time.time() - start_time
    print("\nProcessing complete!")
    print(f"Total time taken: {duration:.2f} seconds")