import cv2
import math

def extract_middle_clip(input_path, output_path, clip_duration_sec, side="right"):
    """
    Extracts a centered clip of the specified duration and side from a stereo video.

    Args:
        input_path (str): Path to the input stereo video (e.g., 640x240).
        output_path (str): Path for the output mono video (e.g., 320x240).
        clip_duration_sec (int): Duration of the clip to extract in seconds (X).
        side (str): Which camera view to extract ('left' or 'right'). Defaults to 'right'.
    """
    # --- 1. Initialize Video Capture and Properties ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        print("Error: Could not read FPS or total frame count.")
        cap.release()
        return

    # Check that the input video width is appropriate for stereo (even number)
    if frame_width % 2 != 0:
        print("Error: Input width is not even. Cannot reliably split.")
        cap.release()
        return

    # --- 2. Calculate Frame Numbers for the Middle Clip ---
    clip_duration_frames = int(clip_duration_sec * fps)
    
    if clip_duration_frames > total_frames:
        print(f"Error: Video is too short ({total_frames / fps:.2f}s) for a {clip_duration_sec}s clip.")
        cap.release()
        return

    # Calculate the start and end frame to center the clip
    start_frame = math.floor((total_frames / 2) - (clip_duration_frames / 2))
    end_frame = start_frame + clip_duration_frames
    start_frame = max(0, start_frame) # Ensure it's not negative

    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Targeting: {side.upper()} view, Frames {start_frame} to {end_frame}")

    # --- 3. Determine Cropping Parameters ---
    new_width = frame_width // 2 # e.g., 640 // 2 = 320
    output_size = (new_width, frame_height)

    # Determine the slice range based on the requested side
    if side.lower() == "left":
        # Left side: columns from 0 up to (but not including) new_width
        col_start, col_end = 0, new_width
    elif side.lower() == "right":
        # Right side: columns from new_width up to the end (frame_width)
        col_start, col_end = new_width, frame_width
    else:
        print("Error: Invalid side specified. Use 'left' or 'right'.")
        cap.release()
        return

    # --- 4. Prepare the Output Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # --- 5. Jump to Start Frame and Process ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_count = start_frame
    
    while cap.isOpened() and current_frame_count < end_frame:
        ret, frame = cap.read()

        if ret:
            # 💡 NumPy Slicing: [All Rows, Columns from col_start to col_end, All Channels]
            cropped_frame = frame[:, col_start:col_end, :]
            
            out.write(cropped_frame)
            current_frame_count += 1
        else:
            break

    # --- 6. Release Resources ---
    cap.release()
    out.release()
    print(f"\n✅ Extracted {clip_duration_sec}-second clip ({side.upper()} view) saved to {output_path}")

# --- Example Usage ---

# 1. Define input/output paths
INPUT_FILE = 'POSE\rear_full.mp4' 
OUTPUT_FILE = '5s_l_rear.mp4'

# 2. Define the parameters
X = 5          # Duration in seconds
VIEW = "left"   # Specify "left" or "right"

extract_middle_clip(INPUT_FILE, OUTPUT_FILE, X, VIEW)