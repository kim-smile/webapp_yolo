import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to the video file
video_path = "youtube_video.mp4"

while True:
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        break

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video. Restarting...")
            break  # Exit inner loop to restart the video

        # Perform detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Get detections

        # Draw bounding boxes and labels
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv5 Object Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Release resources for the current video
    cap.release()

# Clean up
cv2.destroyAllWindows()