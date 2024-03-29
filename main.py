import logging
import os
import time
import gc
import threading
import queue
import traceback

import psutil
import av
import cv2
import numpy as np
from joblib import Parallel, delayed
from moviepy.editor import VideoFileClip
from PIL import Image, ImageStat
from sklearn.neighbors import NearestNeighbors
import tqdm
import ffmpeg
import yaml

###############
### Globals ###
###############

global CORES
global CODEC

def set_globals():
    global CORES
    global CODEC
    try:
        with open("global_cfg.yaml", "r") as f:
            global_config = yaml.safe_load(f)

            CORES = global_config["cores"]
            CODEC = global_config["codec"]

    except Exception as e:
        print(f"An exception occurred while reading global_config.yaml: {e}")
        traceback.print_exc()

        print("Setting default CORES to -1")
        CORES = -1

        print("Setting default CODEC to mp4v")
        CODEC = "mp4v"


###################
### Compilation ###
###################

def save_video(frames_dir, fps, audio_path, output_path, codec):
    """Saves a video from given parameters.

    Args:
        frames_dir (string): The directory where the frames of the video to be saved are contained.
        fps (float): frames per second
        audio_path (string OR None): The path to the audio file for the video audio. If set to None audio will not be added. FFmpeg is required for this to work. Otherwise, a recoverable exception will occur.
        output_path (string): The path where the video will be saved.
        codec (string): fourcc codec used by the cv2 video writer.
    """
    logging.info("Sorting file names.")
    image_filenames = sorted([file for file in os.listdir(frames_dir) if file.endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))

    first_image_path = os.path.join(frames_dir, image_filenames[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)

    logging.info("Initializing video writer.")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_output_path = output_path if not audio_path else output_path.replace(".mp4", "_noaudio.mp4")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, size)

    def write_frames(out, frames_queue):
        while True:
            frame = frames_queue.get()
            if frame is None:
                break
            out.write(frame)

    frames_queue = queue.Queue()

    writer_thread = threading.Thread(target=write_frames, args=(out, frames_queue))
    writer_thread.start()

    logging.info("Writing frames.")
    for filename in tqdm.tqdm(image_filenames):
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)
        frames_queue.put(frame)

    frames_queue.put(None)

    writer_thread.join()
    out.release()

    # will fail if ffmpeg is not installed
    try:
        if audio_path:
            logging.info("Adding audio using ffmpeg-python.")
            (
                ffmpeg
                .input(video_output_path)
                .output(ffmpeg.input(audio_path), output_path, vcodec='copy', acodec='aac', strict='experimental')
                .overwrite_output()
                .run()
            )

            if video_output_path != output_path:
                os.remove(video_output_path)

    except Exception as e:
        print(f"An exception occurred while adding audio: {e}")
        traceback.print_exc()

    logging.info("Done.")


################
### Get Data ###
################

def get_fps(video_path):
    """Returns a video's fps. Uses VideoFileClip(path).fps

    Args:
        video_path (string): Path to get fps.

    Returns:
        float: FPS of the video.
    """
    return VideoFileClip(video_path).fps

def get_audio(video_path):
    """Returns a video's audio.

    Args:
        video_path (string): Path to get audio.

    Returns:
        string OR None: Returns a string if there is a audio, otherwise, an exception will occur and return None.
    """
    try:
        audio_output_path = os.path.splitext(video_path)[0]

        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_output_path)

        audio_clip.close()
        video_clip.close()

        return audio_output_path

    except Exception as e:
        logging.error(f"Error getting audio: {e}")
        return None


##################
### Extraction ###
##################

def video_to_frames(video_path, frames_dir, every=1, scale_factor=1):
    """Outputs every x frames as a .png from a video into a frames directory.

    Args:
        video_path (string): Path to extract frames.
        frames_dir (string): Path to output frames.
        every (int, optional): Every x frame. Defaults to 1.
        scale_factor (int, optional): Scales the frame down by this factor (floor divides dimensions/resolution). Defaults to 1.
    """
    overwrite = False
    start = 0
    end = None

    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)

    assert os.path.exists(video_path)

    os.makedirs(frames_dir, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]

    frame_count = 0
    saved_count = 0

    for frame in tqdm.tqdm(container.decode(stream)):
        if frame_count < start or (end is not None and frame_count >= end) or frame_count % every != 0:
            frame_count += 1
            continue

        img = frame.to_ndarray(format="bgr24")

        # if you dont need to resize, don't do it
        if scale_factor != 1:
            height, width = img.shape[:2]
            new_dimensions = (width // scale_factor, height // scale_factor)
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LANCZOS4)

        save_path = os.path.join(frames_dir, f"{frame_count:010d}.png")
        if overwrite or not os.path.exists(save_path):
            cv2.imwrite(save_path, img)
            saved_count += 1

        frame_count += 1


##############################
### Block Image Generators ###
##############################

class BlockImageGenerator:
    def __init__(self, scale_factor, input_path, output_path, reference_loader, show_progress=False) -> None:
        Image.MAX_IMAGE_PIXELS = None
        self.scale_factor = scale_factor
        self.block_size = reference_loader.block_size
        self.input_path = input_path
        self.output_path = output_path
        self.show_progress = show_progress

        self.reference_tree = reference_loader.reference_tree
        self.reference_names = reference_loader.reference_names
        self.block_dir = reference_loader.block_dir

        self.image = None
        self.image_array = None
        self.indices = None
        self.image_data = None
        self.new_image = None
        self.block_cache = {}

        logging.info("Loading image.")
        self.load_image()

        if self.scale_factor != 1:
            logging.info("Resizing image.")
            self.resize_image()

        else:
            logging.info("Image of good size.")

        logging.info("Making and resizing image array.")
        self.make_then_resize_image_array()

        logging.info("Getting indices.")
        self.get_indices()

        logging.info("Creating image data.")
        self.create_image_data()

        logging.info("Generating block image.")
        self.generate_block_image()

        logging.info("Saving block image.")
        self.save_block_image()

    def load_image(self):
        self.image = Image.open(self.input_path).convert("RGB")

    def resize_image(self):
        new_image_width = self.image.width // self.scale_factor
        new_image_height = self.image.height // self.scale_factor
        self.image = self.image.resize((new_image_width, new_image_height), Image.LANCZOS)

    def make_then_resize_image_array(self):
        self.image_array = np.array(self.image)

        if self.image_array.ndim == 3 and self.image_array.shape[2] > 3:
            self.image_array = self.image_array[:, :, :3]

    def get_indices(self):
        self.indices = self.reference_tree.kneighbors(self.image_array.reshape(-1, 3), return_distance=False)

    def create_image_data(self):
        height, width, _ = self.image_array.shape
        self.image_data = self.reference_names[self.indices[:, 0]].reshape((height, width))

    def generate_block_image(self):
        height, width = self.image_data.shape
        self.new_image = Image.new("RGB", (int(width) * self.block_size, int(height) * self.block_size))

        if self.show_progress:
            for y in tqdm.tqdm(range(height)):
                for x in range(width):
                    block = self.image_data[y, x]

                    if block not in self.block_cache:
                        block_path = os.path.join(self.block_dir, f"{block}.png")
                        self.block_cache[block] = Image.open(block_path).resize((self.block_size, self.block_size), Image.LANCZOS)

                    self.new_image.paste(self.block_cache[block], (x * self.block_size, y * self.block_size))

        else:
            for y in range(height):
                for x in range(width):
                    block = self.image_data[y, x]

                    if block not in self.block_cache:
                        block_path = os.path.join(self.block_dir, f"{block}.png")
                        self.block_cache[block] = Image.open(block_path).resize((self.block_size, self.block_size), Image.LANCZOS)

                    self.new_image.paste(self.block_cache[block], (x * self.block_size, y * self.block_size))

    def save_block_image(self):
        if not self.output_path.endswith(".png"):
            self.output_path += ".png"

        open_cv_image = np.array(self.new_image)
        open_cv_image = open_cv_image[:, :, ::-1]

        cv2.imwrite(self.output_path, open_cv_image)


########################
### Reference Loader ###
########################

class ReferenceLoader:
    def __init__(self, block_dir, allow_transparency=False) -> None:
        self.block_dir = block_dir
        self.allow_transparency = allow_transparency

        self.block_size = 0

        self.block_files = []
        self.loaded_blocks = {}
        self.references = {}

        self.reference_names = None
        self.reference_values = None
        self.reference_tree = None

        logging.info("Getting block files.")
        self.get_block_files()

        logging.info("Getting size of blocks.")
        self.get_size_of_blocks()

        logging.info("Loading blocks.")
        self.load_blocks()

        logging.info("Generating references.")
        self.generate_references()

        logging.info("Generating reference tree.")
        self.generate_keys_and_values()
        self.reshape_reference_values()
        self.generate_reference_tree()

    def get_block_files(self):
        for file_name in os.listdir(self.block_dir):
            if file_name.endswith(".png"):
                self.block_files.append(file_name)

    def get_size_of_blocks(self):
        sizes = set()
        for block_file in self.block_files:
            path_to_block_file = os.path.join(self.block_dir, block_file)
            with Image.open(path_to_block_file) as img:
                sizes.add(img.size)  # img.size is a tuple (width, height)

        logging.info(len(sizes))
        logging.info(sizes)

        if len(sizes) > 1:
            logging.info("Not all block files are the same size. Resorting to smallest value.")
            self.block_size = min(sizes)[0]
        elif len(sizes) == 1:
            self.block_size = next(iter(sizes))[0]
        else:
            raise ValueError("The length of the sizes should not be zero or negative. I don't know how this happened.")

        logging.info(f"Block size to be used is {self.block_size}.")

    def load_blocks(self):
        for block_file in self.block_files:
            path_to_block_file = os.path.join(self.block_dir, block_file)
            block = Image.open(path_to_block_file).convert("RGBA")
            self.loaded_blocks[block_file] = block

    def generate_references(self):
        for block_file, block in self.loaded_blocks.items():
            if not self.allow_transparency:
                alpha_channel = block.split()[-1]
                if alpha_channel.getextrema()[0] < 255: continue

            typeless_file_name = block_file[:-4]

            block = block.convert("RGB")
            block_statistics = ImageStat.Stat(block)

            reference_averages = block_statistics.mean
            references_array = np.array(reference_averages).astype(int)

            self.references[typeless_file_name] = references_array

    def generate_keys_and_values(self):
        self.reference_names = np.array(list(self.references.keys()))
        self.reference_values = np.array(list(self.references.values()))

    def reshape_reference_values(self):
        if self.reference_values.ndim == 1:
            self.reference_values = self.reference_values.reshape(-1, 1)

    def generate_reference_tree(self):
        self.reference_tree = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.reference_values)


######################
### Misc Functions ###
######################

def plot_times(*args):
    pass


############################
### Processing Functions ###
############################

def convert_image(input_path, output_path, reference_loader, scale_factor, show_progress):
    try:
        BlockImageGenerator(input_path=input_path, output_path=output_path, reference_loader=reference_loader, scale_factor=scale_factor, show_progress=show_progress)

    except Exception as e:
        logging.error(f"Error while converting {input_path} to {output_path}: {e}")

def convert_dir(images_dir_path, output_dir_path, reference_loader, scale_factor, show_progress_per_frame, show_progress):
    if not show_progress:
        for image_path in os.listdir(images_dir_path):
            BlockImageGenerator(input_path=os.path.join(images_dir_path, image_path), output_path=os.path.join(output_dir_path, image_path), reference_loader=reference_loader, scale_factor=scale_factor, show_progress=show_progress_per_frame)

    else:
        for image_path in tqdm.tqdm(os.listdir(images_dir_path)):
            BlockImageGenerator(input_path=os.path.join(images_dir_path, image_path), output_path=os.path.join(output_dir_path, image_path), reference_loader=reference_loader, scale_factor=scale_factor, show_progress=show_progress_per_frame)

def dynamic_convert_dir(images_dir_path, output_dir_path, reference_loader, scale_factor=1, show_progress_per_frame=False, decrease_mem_threshold=20, increase_mem_threshold=30, start_batch_size=5, start_adjustment_factor=0.75, cores=-1):
    def process_frame(frame_file):
        start_time = time.perf_counter()
        convert_image(input_path=os.path.join(images_dir_path, frame_file), output_path=os.path.join(output_dir_path, frame_file), reference_loader=reference_loader, scale_factor=scale_factor, show_progress=show_progress_per_frame)
        return time.perf_counter() - start_time

    def get_available_memory_percentage():
        memory = psutil.virtual_memory()
        return 100 * memory.available / memory.total

    def adjust_batch_size(current_batch_size, adjustment_factor):
        available_memory = get_available_memory_percentage()

        # Dynamically adjust batch size based on memory usage
        if available_memory < decrease_mem_threshold:
            new_batch_size = max(1, int(current_batch_size * adjustment_factor))
        elif available_memory > increase_mem_threshold:
            new_batch_size = int(current_batch_size / adjustment_factor)
        else:
            new_batch_size = current_batch_size

        return new_batch_size

    image_files = sorted(os.listdir(images_dir_path))
    current_batch_size = start_batch_size
    adjustment_factor = start_adjustment_factor
    frame_processing_times = []
    processed_images = 0

    while processed_images < len(image_files):
        logging.info("Calculating batch information")
        batch_end = min(processed_images + current_batch_size, len(image_files))
        current_batch_files = image_files[processed_images:batch_end]

        logging.info("Working.")
        batch_processing_times = Parallel(n_jobs=cores)(delayed(process_frame)(image_file) for image_file in tqdm.tqdm(current_batch_files))

        logging.info("Extending frame processing times")
        frame_processing_times.extend(batch_processing_times)
        processed_images += len(current_batch_files)

        # Dynamically adjust batch size based on system memory usage
        logging.info("Adjusting batch size")
        current_batch_size = adjust_batch_size(current_batch_size, adjustment_factor)

        # Explicit garbage collection to free up memory
        logging.info("Collecting garbage")
        gc.collect()

        logging.info(f"{processed_images} / {len(image_files)} complete")

    return frame_processing_times

def convert_video(video_path, output_path, reference_loader, scale_factor, frames_dir="frames_dir", clean_up=True, benchmark=False, decrease_mem_threshold=20, increase_mem_threshold=30, start_batch_size=5, start_adjustment_factor=0.75):
    global CORES
    global CODEC

    start_time = time.perf_counter()

    vtf_start_time = time.perf_counter()
    video_to_frames(video_path=video_path, frames_dir=frames_dir, scale_factor=scale_factor)
    vtf_end_time = time.perf_counter()

    fp_start_time = time.perf_counter()
    frame_processing_times = dynamic_convert_dir(images_dir_path=frames_dir, output_dir_path=frames_dir, reference_loader=reference_loader, show_progress_per_frame=False,
                                                 decrease_mem_threshold=decrease_mem_threshold, increase_mem_threshold=increase_mem_threshold, start_batch_size=start_batch_size,
                                                 start_adjustment_factor=start_adjustment_factor, cores=CORES)
    fp_end_time = time.perf_counter()

    sv_start_time = time.perf_counter()
    save_video(frames_dir, get_fps(video_path), get_audio(video_path + "_audio.mp3"), output_path, CODEC)
    sv_end_time = time.perf_counter()

    clean_start_time = time.perf_counter()
    if clean_up:
        for frame_path in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, frame_path))
        os.removedirs(frames_dir)
        os.remove(video_path + "_audio.mp3")
    clean_end_time = time.perf_counter()

    end_time = time.perf_counter()

    data = {
        "VideoPath": video_path,
        "FrameRate": get_fps(video_path),
        "VideoLength": cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT) /  get_fps(video_path),

        "ScaleFactor": scale_factor,
        "InitialResolution": (int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH)), int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "OutputResolution": (int((cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH))/scale_factor*reference_loader.block_size), (int((cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT))/scale_factor*reference_loader.block_size))),

        "OutputPath": output_path,
        "FramesDir": frames_dir,
        "CPUBool": True,
        "BenchmarkMode": benchmark,

        "VideoToFrames": vtf_end_time-vtf_start_time,
        "FrameProcessing": fp_end_time-fp_start_time,
        "SavingVideo": sv_end_time-sv_start_time,
        "CleaningUp": clean_end_time-clean_start_time,
        "DidCleanUp": clean_up,

        "AverageFrameTime": sum(frame_processing_times) / len(frame_processing_times),

        "DecreaseMemoryThreshold": decrease_mem_threshold,
        "IncreaseMemoryThreshold": increase_mem_threshold,
        "StartBatchSize": start_batch_size,
        "StartAdjustmentFactor": start_adjustment_factor,

        "StartTime": start_time,
        "EndTime": end_time,
        "Elapsed": end_time-start_time,
    }

    if not benchmark:
        for point in data:
            print(f"{point}= {data[point]}")  # print, not logging.info for formatting purposes

        plot_times(frame_processing_times)

    else:
        with open(f"{output_path}_benchmark.txt", "w") as f:
            for point in data:
                f.write(f"{point}= {data[point]}\n")

    return data


#########################
### Running Functions ###
#########################

# definitely repeated myself...
# numerous times

def config_convert_image(reference_loader, scale_factor):
    input_path = input("Image input path: ").strip()
    if not os.path.exists(input_path):
        logging.error(f"Path '{input_path}' does not exist. Make sure you typed in the proper location.")
        exit()

    output_path = input("Image output path: ").strip()

    if not output_path.endswith(".png"):
        logging.warning("Your output path does not end with '.png'.")
        add_png_input = input("Add '.png'? (Y/N): ")
        match add_png_input.upper():
            case "Y":
                output_path += ".png"

            case "N":
                pass  # do nothing

            case _:
                print("Not 'Y' or 'N'. Resorting to No.")
                pass  # do nothing

    logging.info(f"Absolute path of {output_path} is {os.path.abspath(output_path)}")
    if os.path.exists(os.path.abspath(output_path)):
        logging.warning(f"Path '{os.path.abspath(output_path)}' already exists. Continuing this program will overwrite the preexisting file.")
        input("Press enter to continue, or cancel execution.")

    show_progress_input = input("Show progress? (Y/N): ")
    match show_progress_input.upper():
        case "Y":
            show_progress = True

        case "N":
            show_progress = False

        case _:
            print("Not 'Y' or 'N'. Resorting to Yes.")
            show_progress = True

    convert_image(input_path=input_path, output_path=output_path, reference_loader=reference_loader, scale_factor=scale_factor, show_progress=show_progress)

def config_convert_dir(reference_loader, scale_factor):
    global CORES

    input_path = input("Directory input path: ").strip()
    if not os.path.exists(input_path):
        logging.error(f"Path '{input_path}' does not exist. Make sure you typed in the proper location.")
        exit()

    output_path = input("Directory output path: ").strip()
    logging.info(f"Absolute path of {output_path} is {os.path.abspath(output_path)}")
    if not os.path.exists(os.path.abspath(output_path)):
        logging.warning(f"Path '{os.path.abspath(output_path)}' does not exist. Continuing will create the directory.")
        input("Press enter to continue, or cancel execution.")
        os.mkdir(os.path.abspath(output_path))

    elif os.listdir(os.path.abspath(output_path)):
        logging.warning(f"Path '{os.path.abspath(output_path)}' has content in it. Continuing will add to the directory.")
        input("Press enter to continue, or cancel execution.")

    show_progress_per_frame_input = input("Show progress per image in directory? (Y/N): ")
    match show_progress_per_frame_input.upper():
        case "Y":
            show_progress_per_frame = True

        case "N":
            show_progress_per_frame = False

        case _:
            print("Not 'Y' or 'N'. Resorting to No.")
            show_progress_per_frame = False

    show_progress_input = input("Show progress for entire directory? (Y/N): ")
    match show_progress_input.upper():
        case "Y":
            show_progress = True

        case "N":
            show_progress = False

        case _:
            print("Not 'Y' or 'N'. Resorting to Yes.")
            show_progress = False

    dynamic_convert_input = input("Dynamically convert the directory? (Y/N): ")
    match dynamic_convert_input.upper():
        case "Y":
            dynamic_convert = True

        case "N":
            dynamic_convert = False

        case _:
            print("Not 'Y' or 'N'. Resorting to No.")
            dynamic_convert = False

    if not dynamic_convert:
        convert_dir(images_dir_path=os.path.abspath(input_path), output_dir_path=os.path.abspath(output_path), reference_loader=reference_loader, scale_factor=scale_factor, show_progress_per_frame=show_progress_per_frame, show_progress=show_progress)

    else:
        print("Dynamic directory conversion automatically uses multiple CPU cores it can (via joblib) to run faster.")
        print("It also has four configuration options in order to optimize memory usage when working with large amounts of images at the same time. (Batch processing)")

        decrease_mem_threshold_input = input("Memory percentage left to throttle (Default- 20): ")
        try:
            decrease_mem_threshold = int(decrease_mem_threshold_input)
        except ValueError:
            logging.error(f"Memory percentage left to throttle {decrease_mem_threshold_input} is not an integer.")
            exit()

        increase_mem_threshold_input = input("Memory percentage used to increase (Default- 30): ")
        try:
            increase_mem_threshold = int(increase_mem_threshold_input)
        except ValueError:
            logging.error(f"Memory percentage used to increase {increase_mem_threshold_input} is not an integer.")
            exit()

        start_batch_size_input = input("Start batch size (Default- 5): ")
        try:
            start_batch_size = int(start_batch_size_input)
        except ValueError:
            logging.error(f"Start batch size {start_batch_size_input} is not an integer.")
            exit()

        start_adjustment_factor_input = input("Start adjustment factor (Default- 0.75): ")
        try:
            start_adjustment_factor = float(start_adjustment_factor_input)
        except ValueError:
            logging.error(f"Start adjustment factor {start_adjustment_factor_input} is not a float.")
            exit()

        dynamic_convert_dir(images_dir_path=os.path.abspath(input_path), output_dir_path=os.path.abspath(output_path), reference_loader=reference_loader, scale_factor=scale_factor, show_progress_per_frame=show_progress_per_frame,
                            decrease_mem_threshold=decrease_mem_threshold, increase_mem_threshold=increase_mem_threshold, start_batch_size=start_batch_size, start_adjustment_factor=start_adjustment_factor, cores=CORES)

def config_convert_video(reference_loader, scale_factor):
    input_path = input("Video input path: ").strip()
    if not os.path.exists(input_path):
        logging.error(f"Path '{input_path}' does not exist. Make sure you typed in the proper location.")
        exit()

    output_path = input("Video output path: ").strip()

    if not output_path.endswith(".mp4"):
        logging.warning("Your output path does not end with '.mp4'")
        add_png_input = input("Add '.mp4'? (Y/N): ")
        match add_png_input.upper():
            case "Y":
                output_path += ".mp4"

            case "N":
                pass  # do nothing

            case _:
                print("Not 'Y' or 'N'. Resorting to No.")
                pass  # do nothing

    logging.info(f"Absolute path of {output_path} is {os.path.abspath(output_path)}")
    if os.path.exists(os.path.abspath(output_path)):
        logging.warning(f"Path '{os.path.abspath(output_path)}' already exists. Continuing this program will overwrite the preexisting file.")
        input("Press enter to continue, or cancel execution.")

    print("The frames output path (frames dir) is a temporary directory that will be deleted at the end of execution.")
    frames_directory = input("Frames output path: ").strip()
    if not frames_directory:
        print("Empty input. Resorting to 'frames_dir'")
        frames_directory = "frames_dir"

    logging.info(f"Absolute path of {frames_directory} is {os.path.abspath(frames_directory)}")
    if os.path.exists(os.path.abspath(output_path)) and os.listdir(os.path.abspath(output_path)):
        logging.warning(f"Path '{os.path.abspath(output_path)}' has content in it. Continuing will add to the directory. All content in this directory will be deleted at the end of execution.")
        input("Press enter to continue, or cancel execution.")

    print("Video conversion automatically uses dynamic directory conversion.")
    print("Dynamic directory conversion automatically uses all CPU cores it can (via joblib) to run faster.")
    print("It also has four configuration options in order to optimize memory usage when working with large amounts of images at the same time. (Batch processing)")

    decrease_mem_threshold_input = input("Memory percentage left to throttle (Default- 20): ")
    try:
        decrease_mem_threshold = int(decrease_mem_threshold_input)
    except ValueError:
        logging.error(f"Memory percentage left to throttle {decrease_mem_threshold_input} is not an integer.")
        exit()

    increase_mem_threshold_input = input("Memory percentage used to increase (Default- 30): ")
    try:
        increase_mem_threshold = int(increase_mem_threshold_input)
    except ValueError:
        logging.error(f"Memory percentage used to increase {increase_mem_threshold_input} is not an integer.")
        exit()

    start_batch_size_input = input("Start batch size (Default- 5): ")
    try:
        start_batch_size = int(start_batch_size_input)
    except ValueError:
        logging.error(f"Start batch size {start_batch_size_input} is not an integer.")
        exit()

    start_adjustment_factor_input = input("Start adjustment factor (Default- 0.75): ")
    try:
        start_adjustment_factor = float(start_adjustment_factor_input)
    except ValueError:
        logging.error(f"Start adjustment factor {start_adjustment_factor_input} is not a float.")
        exit()

    convert_video(video_path=os.path.abspath(input_path), output_path=os.path.abspath(output_path), reference_loader=reference_loader, scale_factor=scale_factor, frames_dir=os.path.abspath(frames_directory),
                  decrease_mem_threshold=decrease_mem_threshold, increase_mem_threshold=increase_mem_threshold, start_batch_size=start_batch_size, start_adjustment_factor=start_adjustment_factor)

def change_block_dir(reference_loader: ReferenceLoader):
    block_dir = input("Reference loader block directory: ").strip()

    if not os.path.exists(block_dir):
        logging.error(f"Path '{block_dir}' does not exist. Make sure you typed in the proper location.")
        exit()

    allow_transparency_input = input("Allow transparency (Y/N): ").strip()
    match allow_transparency_input.upper():
        case "Y":
            allow_transparency = True

        case "N":
            allow_transparency = False

        case _:
            print("Not 'Y' or 'N'. Resorting to No.")
            allow_transparency = False

    return ReferenceLoader(block_dir=block_dir, allow_transparency=allow_transparency)

def change_block_size(reference_loader: ReferenceLoader):
    block_size_input = input("New block size: ")
    try:
        block_size = int(block_size_input)

    except ValueError:
        logging.error(f"Start batch size {block_size_input} is not an integer.")
        exit()

    reference_loader.block_size = block_size

def change_scale_factor():
    print("Note: Blockvideo Studio handles scale factors as scale 'divisors'. Scale factors are integers that act as the divisor to the original resolution.")
    scale_factor_input = input("Scale factor: ")
    try:
        sf = int(scale_factor_input)

    except ValueError:
        logging.error(f"Scale factor {scale_factor_input} is not an integer.")
        change_scale_factor()

    return sf


############
### Main ###
############

def main():
    try:
        logging.basicConfig(level=logging.INFO)

        print("All images/videos produced by Blockvideo Studio are uncompressed and may take up large file sizes.\nThere are no checks in place to prevent large files or directories from being generated.")

        block_dir = input("Reference loader block directory: ").strip()

        if not os.path.exists(block_dir):
            logging.error(f"Path '{block_dir}' does not exist. Make sure you typed in the proper location.")
            exit()

        allow_transparency_input = input("Allow transparency (Y/N): ").strip()
        match allow_transparency_input.upper():
            case "Y":
                allow_transparency = True

            case "N":
                allow_transparency = False

            case _:
                print("Not 'Y' or 'N'. Resorting to No.")
                allow_transparency = False

        rf = ReferenceLoader(block_dir=block_dir, allow_transparency=allow_transparency)

        print("Note: Blockvideo Studio handles scale factors as scale 'divisors'. Scale factors are integers that act as the divisor to the original resolution.")
        scale_factor_input = input("Scale factor: ")
        try:
            sf = int(scale_factor_input)

        except ValueError:
            logging.error(f"Scale factor {scale_factor_input} is not an integer.")
            change_scale_factor()

        while True:
            try:
                print(f"Block directory is {rf.block_dir}")
                print(f"Block size is {rf.block_size}")
                print(f"Scale factor is {sf}")

                print("""Choose option:
    1) Convert Single Image
    2) Convert Directory
    3) Convert Video
    4) Remake Reference Loader (Change Block Directory)
    5) Change Block Size
    6) Change Scale Factor
            """)
                option_input = input(": ").strip()
                match option_input:
                    case "1": config_convert_image(rf, sf)
                    case "2": config_convert_dir(rf, sf)
                    case "3": config_convert_video(rf, sf)
                    case "4": rf = change_block_dir(rf)
                    case "5": change_block_size(rf)
                    case "6": sf = change_scale_factor()
                    case _: print("Invalid option.")

                print("Finished.")

            except Exception as e:
                print(f"An exception occurred in the loop: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"An exception occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    set_globals()
    main()
