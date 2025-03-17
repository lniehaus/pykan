import os
import numpy as np
import moviepy.video.io.ImageSequenceClip # moviepy == 1.0.3

def create_video(image_folder, video_name='training', fps=5):
    # video_name='video'
    # fps=5

    #fps = fps
    files = os.listdir(image_folder)
    train_index = []
    for file in files:
        if file[0].isdigit() and file.endswith('.jpg'):
            train_index.append(int(file[:-4]))

    train_index = np.sort(train_index)

    image_files = [image_folder+'/'+str(train_index[index])+'.jpg' for index in train_index]

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    video_file = os.path.join(image_folder, video_name+'.mp4')
    clip.write_videofile(video_file)
