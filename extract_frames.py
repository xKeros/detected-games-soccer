import cv2
import os

def extract_frames(video_path, output_folder, fps=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = int(frame_rate / fps)
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            filename = f"{output_folder}/{video_name}_frame{frame_id}.jpg"
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1

    cap.release()
    print(f"Extracción de fotogramas completada para {video_name}.")

# Ejemplo de uso
if __name__ == "__main__":
    video_folder = "./dataset/videos"
    output_folder = "./dataset/frames"
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, output_folder)
