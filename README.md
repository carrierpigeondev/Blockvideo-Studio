# Blockvideo Studio
Blockvideo Studio is a CLI program to convert images, directories of images, and videos into Blockimages / Blockvideos, images and videos made up of 'blocks' (other individual images), by replacing each pixel in an image or video with another image.

# Examples
https://imgur.com/5yeBVeM from [u/zlakphoto](https://www.reddit.com/user/zlakphoto/)
![Original Image](examples/5yeBVeM.jpeg)

Block Directory: `examples/nature_animals` (Split up iOS/MacOS Emojis)  
Block Size: 50 (50x50 images)  
Scale Factor: 50

![Blockimage](examples/5yeBVeM_converted.png)

---
https://i.redd.it/u5p5b3twe3251.jpg from [u/Yoredlol](https://www.reddit.com/user/Yoredlol/)
![Original Image](examples/u5p5b3twe3251.jpg)

Blocks directory: `examples/POLA5` (Source: https://lospec.com/palette-list/pola5)  
Block Size: 4  
Scale Factor: 4

![Blockimage](examples/u5p5b3twe3251_converted.png)

---

https://archive.org/details/bliss-600dpi from [Charles O'Rear](https://archive.org/search?query=creator%3A%22Charles+O%27Rear%22)
![Original Image](examples/bliss-600dpi.png)

Blocks Directory: `examples/Win95` (Source: https://lospec.com/palette-list/windows-95-256-colours)  
Block Size: 50  
Scale Factor: 60

![Original Image](examples/bliss-600dpi_converted.png)

# Prerequisites
1. Python 3.12* and pip
2. FFmpeg

\* older versions may work, but haven't been tested

# Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run the CLI program: `python .\main.py` or `py .\main.py`
