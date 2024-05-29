import cv2
import numpy as np
import os

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera, or provide the video file path

# Initialize the reference frame
ref_frame = None

# Threshold for detecting significant changes
threshold = 150000

while True:
    # Read a frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur the frame to reduce noise
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Set the reference frame if it's not set yet
    if ref_frame is None:
        ref_frame = gray_frame
        continue

    # Compute the absolute difference between the current frame and the reference frame
    frame_delta = cv2.absdiff(ref_frame, gray_frame)

    # Threshold the frame delta to get binary image
    _, thresh_delta = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels in the thresholded image
    non_zero_count = np.sum(thresh_delta != 0)

    # Check if the number of non-zero pixels exceeds the threshold
    if non_zero_count > threshold:
        print("Significant Motion Detected!")
        # Play system notification sound
        os.system("afplay /System/Library/Sounds/Ping.aiff")  # macOS
        # For Windows: winsound.Beep(500, 500)  # Adjust frequency and duration as needed

    # Update the reference frame
    ref_frame = gray_frame

    # Display the original frame and the frame delta
    cv2.imshow('Original', frame)
    cv2.imshow('Frame Delta', frame_delta)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
