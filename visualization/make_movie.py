import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = 'movie'
ext = args['extension']
output = args['output']

FPS = 60.0
n_images = 10000

images = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
images = [[x, os.path.getmtime(x)] for x in images]
images = sorted(images, key=lambda t: t[-1])
images = [x[0] for x in images]
images = images[:n_images]

# Determine the width and height from the first image
image_path = images[0]
frame = cv2.imread(image_path)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, FPS, (width, height))

for image in images:

    image_path = image
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    #cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
