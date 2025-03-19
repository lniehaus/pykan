import os
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image

# import moviepy.video.io.ImageSequenceClip # moviepy == 1.0.3

# def create_video(image_folder, video_name='training', fps=5):

#     #fps = fps
#     files = os.listdir(image_folder)
#     train_index = []
#     for file in files:
#         if file[0].isdigit() and file.endswith('.jpg'):
#             train_index.append(int(file[:-4]))

#     train_index = np.sort(train_index)

#     image_files = [image_folder+'/'+str(train_index[index])+'.jpg' for index in train_index]

#     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
#     video_file = os.path.join(image_folder, video_name+'.mp4')
#     clip.write_videofile(video_file)

#     # video_file = os.path.join(image_folder, video_name+'.mp4')
#     # clip.write_videofile(video_file, codec='libx264')
#     # video_file = os.path.join(image_folder, video_name+'.avi')
#     # clip.write_videofile(video_file, codec='libxvid')
#     # video_file = os.path.join(image_folder, video_name+'.webm')
#     # clip.write_videofile(video_file, codec='libvpx')
#     return video_file



# def create_video(image_folder, video_name='training', fps=5):
#     # Get a list of image files
#     files = os.listdir(image_folder)
#     train_index = []
    
#     for file in files:
#         if file[0].isdigit() and file.endswith('.jpg'):
#             train_index.append(int(file[:-4]))

#     train_index = np.sort(train_index)

#     image_files = [os.path.join(image_folder, f"{index}.jpg") for index in train_index]

#     # Create a video writer object
#     video_file = os.path.join(image_folder, f"{video_name}.mp4")
#     with imageio.get_writer(video_file, fps=fps) as writer:
#         for image_file in tqdm(image_files):
#             image = imageio.imread(image_file)
#             writer.append_data(image)

#     return video_file



def create_video(image_folder, video_name='training', fps=5):
    # Get a list of image files
    files = os.listdir(image_folder)
    train_index = []
    
    for file in files:
        if file[0].isdigit() and file.endswith('.jpg'):
            train_index.append(int(file[:-4]))

    train_index = np.sort(train_index)

    image_files = [os.path.join(image_folder, f"{index}.jpg") for index in train_index]

    # Create a video writer object
    video_file = os.path.join(image_folder, f"{video_name}.mp4")
    with imageio.get_writer(video_file, fps=fps) as writer:
        for image_file in tqdm(image_files):
            image = Image.open(image_file)

            # Resize the image to be divisible by 16
            width, height = image.size
            new_width = (width // 16) * 16
            new_height = (height // 16) * 16
            image = image.resize((new_width, new_height), Image.LANCZOS)

            writer.append_data(np.array(image))

    return video_file