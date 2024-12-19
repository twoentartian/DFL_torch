import os
import cv2
import argparse
from PIL import Image


def generate_video_and_gif(folder, range_arg, video_output, gif_output, fps=10):
    # Parse the range argument
    try:
        start, stop, step = map(int, range_arg.split(':'))
    except ValueError:
        print("Invalid range argument format. Use 'start:stop:step'")
        return

    # Collect selected image paths
    # Collect image filenames and extract numeric parts
    images = []
    for f in os.listdir(folder):
        if f.endswith('.png'):
            try:
                num = int(os.path.splitext(f)[0])
                images.append((num, os.path.join(folder, f)))
            except ValueError:
                continue  # Skip files that do not have numeric names

    # Sort images by their numeric value
    images.sort(key=lambda x: x[0])

    # Select images based on the provided range
    selected_images = [img[1] for img in images if start <= img[0] < stop and (img[0] - start) % step == 0]

    if not selected_images:
        print("No images selected based on the provided range.")
        return

    print(selected_images)

    # Create a video
    frame = cv2.imread(selected_images[0])
    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_path in selected_images:
        frame = cv2.imread(img_path)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as {video_output}")

    # Create a GIF
    frames = [Image.open(img_path) for img_path in selected_images]
    frames[0].save(gif_output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"GIF saved as {gif_output}")


def main():
    parser = argparse.ArgumentParser(description="Generate a video and GIF from selected images.")
    parser.add_argument("range_arg",type=str,help="Range of images to include in the format 'start:stop:step'.")
    parser.add_argument("--folder",type=str,default="video_cache",help="Folder containing the image files. Default is 'video_cache'.")
    parser.add_argument("--video_output",type=str,default="output_video.mp4",help="Output video file name. Default is 'output_video.mp4'.")
    parser.add_argument("--gif_output",type=str,default="output.gif",help="Output GIF file name. Default is 'output.gif'.")
    parser.add_argument("--fps",type=int,default=2,help="Frames per second for the video and GIF. Default is 10.")

    args = parser.parse_args()
    generate_video_and_gif(args.folder, args.range_arg, args.video_output, args.gif_output, args.fps)


if __name__ == "__main__":
    main()
