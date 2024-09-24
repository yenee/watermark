import cv2
import os
import glob
import numpy as np

# Function to create a rotated watermark image with alpha channel
def create_rotated_watermark(text, angle, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(100, 100, 100), thickness=2, transparency=0.3):
    # Create a blank image to get text size
    temp_img = np.zeros((1, 1, 3), dtype=np.uint8)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Create a larger image to accommodate rotated text
    max_dim = int(np.sqrt(text_w**2 + text_h**2)) + 10  # Add padding
    text_img = np.zeros((max_dim, max_dim, 4), dtype=np.uint8)  # 4 channels: BGR + Alpha
    
    # Draw text on the image
    text_img = cv2.putText(text_img, text, (max_dim // 2 - text_w // 2, max_dim // 2 + text_h // 2), font, font_scale, color + (255,), thickness)

    # Rotate text image
    center = (max_dim // 2, max_dim // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_text_img = cv2.warpAffine(text_img, rotation_matrix, (max_dim, max_dim), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Apply transparency
    alpha_channel = rotated_text_img[:, :, 3] / 255.0 * transparency
    rotated_text_img[:, :, 3] = (alpha_channel * 255).astype(np.uint8)  # Update alpha channel with adjusted transparency

    return rotated_text_img


def add_watermarks_to_video(video_path, output_path, watermark_text, width_num=3, height_num=3):
    # Open video
    cap = cv2.VideoCapture(video_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Parameters for watermarks
    angle = 25  # Incline angle
    distance_x = width // width_num  # Horizontal distance between watermarks
    distance_y = height // height_num  # Vertical distance between watermarks
    margin = 50  # Margin from the edges
    
    # Calculate positions for watermarks once
    positions = []
    x_start = margin
    y_start = margin
    stagger_offset = distance_x // 2  # Offset for staggering
    
    # Create the watermark image
    watermark_img = create_rotated_watermark(watermark_text, angle)
    watermark_h, watermark_w = watermark_img.shape[:2]
    line_index = 0  # To alternate the x position

    while y_start < height - margin:
        if line_index % 2 == 0:
            x_start = margin
        else:
            x_start = stagger_offset
        
        x = x_start
        while x < width - margin:
            positions.append((x, y_start))
            x += distance_x
        
        y_start += distance_y
        line_index += 1

    frame_count = 0
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add watermark text to the frame
        for pos in positions:
            x, y = pos
            x_start = max(x - watermark_w // 2, 0)
            y_start = max(y - watermark_h // 2, 0)
            x_end = min(x_start + watermark_w, width)
            y_end = min(y_start + watermark_h, height)
            
            # Extract region from watermark image
            watermark_region = watermark_img[:y_end - y_start, :x_end - x_start]
            alpha = watermark_region[:, :, 3] / 255.0
            overlay = watermark_region[:, :, :3]

            # Overlay the watermark onto the frame
            for c in range(3):
                frame[y_start:y_end, x_start:x_end, c] = (1 - alpha) * frame[y_start:y_end, x_start:x_end, c] + alpha * overlay[:, :, c]

        # Write frame to output video
        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Processed frame {frame_count}')

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_videos_in_folder(folder_path, watermark_text, width_num=3, height_num=3):
    # Find all MP4 files in the folder
    mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
    output_folder = os.path.join(folder_path, 'watermarked_videos')
    os.makedirs(output_folder, exist_ok=True)
    # Process each video file
    for video_path in mp4_files:
        output_path = os.path.join(output_folder, os.path.basename(video_path))
        print(f'----- Processing {video_path} -----')
        add_watermarks_to_video(video_path, output_path, watermark_text, width_num, height_num)
        print(f'----- Finished {video_path} -----')

folder_path = os.getcwd() 
watermark_text = input('Enter the watermark text: ') 
width_num = 3
height_num = 3

process_videos_in_folder(folder_path, watermark_text, width_num, height_num)
