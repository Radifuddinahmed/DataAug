    # input_loc = '/LPBF/1/video/aav4687s1.mov'
    # output_loc = '/LPBF/1/frames/'


import cv2

# Open the video file
video_path = 'aav4687s1.MOV'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a directory to save the frames (if it doesn't exist)
import os
output_dir = 'LPBF/1/frames/'  # Replace with your desired output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize variables
frame_count = 0

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image file (e.g., PNG or JPEG)
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)

    # Increment the frame count
    frame_count += 1

# Release the video file
cap.release()

print(f"Frames extracted: {frame_count}")

vid = cv2.VideoCapture( 'aav4687s1.MOV')
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # always 0 in Linux python3
width  = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # always 0 in Linux python3
print ("opencv: height:{} width:{}".format( height, width))
