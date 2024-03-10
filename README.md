# Blockvideo Studio
Blockvideo Studio is a CLI program to convert images, directories of images, and videos into Blockimages / Blockvideos, images and videos made up of 'blocks' (other individual images), by replacing each pixel in an image or video with another image.

# Example
https://imgur.com/5yeBVeM from [u/zlakphoto](https://www.reddit.com/user/zlakphoto/)
![Original Image](examples\5yeBVeM.jpeg)

Using the `examples\cold_palette\` directory for blocks, with a scale factor of 1, we get this output:

![Blockimage](examples\5yeBVeM_converted.png)



# Prerequisites
1. Python 3.12* and PIP
2. FFmpeg

\* older versions may work, but haven't been tested

# Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run the CLI program: `python .\main`