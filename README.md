# Blockvideo Studio
Blockvideo Studio is a CLI program to convert images, directories of images, and videos into Blockimages / Blockvideos, images and videos made up of 'blocks' (other individual images), by replacing each pixel in an image or video with another image.

# Examples
https://imgur.com/5yeBVeM from [u/zlakphoto](https://www.reddit.com/user/zlakphoto/)
![Original Image](examples/5yeBVeM.jpeg)

Using the `examples/cold_palette/` directory for blocks, with a scale factor of 1, we get this output:

![Blockimage](examples/5yeBVeM_converted.png)

---

https://i.redd.it/v6i2u32nxu751.jpg from [u/matiorex](https://www.reddit.com/user/Matiorex/)
![Original Image](examples/v6i2u32nxu751.jpg)

Using the `examples/25_palette/` directory for blocks, with a scale factor of 6 and a modified block size of 6, we get this output:

![Blockimage](examples/v6i2u32nxu751_converted.png)

---
https://i.redd.it/u5p5b3twe3251.jpg from [u/Yoredlol](https://www.reddit.com/user/Yoredlol/)
![Original Image](examples/u5p5b3twe3251.jpg)

Using the `examples/nature_animals` directory for blocks, with a scale factor of 50, the block size is 50 (50x50 resolution images), and we get this output:

![Blockimage](examples/u5p5b3twe3251_converted.png)

# Prerequisites
1. Python 3.12* and PIP
2. FFmpeg

\* older versions may work, but haven't been tested

# Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run the CLI program: `python .\main.py`
