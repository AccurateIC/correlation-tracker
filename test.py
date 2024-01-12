import cv2

def show_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and display frames until the user presses 'q' or the video ends
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame is not read successfully, break the loop
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_file = r'/home/accurate2/pyCFT/pyCFTrackers/output_videos/van_output.mp4'
show_video(video_file)
