from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path, audio_output):
    clip = VideoFileClip(video_path)
    audio_path = os.path.join(audio_output, os.path.basename(video_path).replace('.mp4', '.wav'))
    clip.audio.write_audiofile(audio_path)
    print(f"Audio extraído: {audio_path}")

# Ejemplo de uso
if __name__ == "__main__":
    video_folder = "./dataset/videos"
    audio_output = "./dataset/audio"
    if not os.path.exists(audio_output):
        os.makedirs(audio_output)
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        extract_audio(video_path, audio_output)
