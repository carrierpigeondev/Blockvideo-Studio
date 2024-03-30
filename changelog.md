# Version 1

## 1.1.11 (To be added with next commit)
Fixed
* Fixed audio path issues that were first caused in an earlier version
* Fixed issue when trying to convert a video and save to a path that already exists

Added
* changelog.md to keep track of changes

## 1.1.10 (2024-3-29 17:54 CDT)
Added
* More notes to the README

## 1.1.9 (2024-3-29 17:20 CDT)
Added
* Numerous more things in pylint disable

Changed
* Now uses .items() in for loop for printing data after running video conversion

Fixed
* Fixed arguments in match-case in the option selection (Fixed the fix)

## 1.1.8 (2024-3-29 17:13 CDT)
Fixed
* Fixed arguments in match-case in the option selection (Didn't fix)

## 1.1.7 (2024-3-29 17:04 CDT)
Added
* Numerous more things in pylint disable
* Specified encoding type in with open()s

Removed
* Removed ChatGPT's non-existant pylint options
* Unnecessary reference_loader parameter in change_block_dir

Changed
* Unused layers variable to _

## 1.1.6 (2024-3-29 16:42 CDT)
Added
* logging-fstring-interpolation to pylint disable

Changed
* pylint now ignores cv2 and PIL as would give false warnings
* Direct point to the rcfile .pylintrc

Fixed
* Changed all exit() to sys.exit()

## 1.1.5 (2024-3-29 16:28 CDT)
Fixed
* pylint now installs dependencies from requirements.txt

## 1.1.4 (2024-3-29 16:25 CDT)
Changed
* Increased max line length for pylint

Fixed
* Removed all trailing whitespace lines

## 1.1.3 (2024-3-29 16:15 CDT)
Added
* ChatGPT generated .pylinrc

Fixed
* Indentation inconsitencies and some whitespace lines

### 1.1.2 (2024-3-29 16:02 CDT)
Commit: e45fe3b579b1f77f9763db4f612416dab8d97405

### 1.1.1 (2024-3-29 16:02 CDT)
Changes
* Added a new line, separating the Globals and Compilation section to the right amount; mainly done to test pylint

### 1.1.0 (2024-3-29 15:59 CDT)
Added
* Implemented pylint through GitHub Actions

## 1.0 - Public Repository

### 1.0.7 (2024-3-29 15:41 CDT)
Fixed
* Added a call to set_globals(), where it didn't get called before

### 1.0.6 (2024-3-29 15:41 CDT)
Fixed
* Fixed numerous issues with globals

### 1.0.5 (2024-3-29 15:33 CDT)
Added
* Docstrings to non-proprietary functions
* Removed globals in non-proprietary functions, added them as parameters, passed as arguments when called in proprietary functions
* Clean up video file

### 1.0.4 (2024-3-29 15:12 CDT)
Added
* .gitignore
* global_cfg.yaml and global configuration options
* * Added more details to the prerequisites
* Added pyyaml to requirements.txt

Changed
* Changed a few print() to logging.info() in certain situations
* Added try-except to adding audio to catch if FFmpeg is not installed
* Changed wording in some print() and logging.info()

***-- took a break after leaving vegas :P***

### 1.0.3 (2024-3-13 22:29 CDT)
Added
* Temporary ChatGPT generated documentation on usage and code explanation

### 1.0.2 (2024-3-13 21:56 CDT)
Commit 89b81d2f54053c5c5325fdc13b1f644d2ff2e455

### 1.0.1 (2024-3-13 21:56 CDT)
Changed
* Changed default to yes to show progress when converting an image

### 1.0.0 (2024-3-13 19:11 CDT)
Added
* GPL-3.0 license

# Pre-1.0

## 0.1 - Revamping Examples

### 0.1.2 (2024-3-13 18:47 CDT)
Added
* Added another example -> Windows XP Bliss with Windows 95 Colors :)

### 0.1.1 (2024-3-13 18:18 CDT)
Changed
* / to \ for an example in the README
* Uncapitalized "PIP" to "pip" in the README
* Changed indentation level of the option choosing in the CLI

Added
* Added "or `py .\main.py" to the README

### 0.1.0 (2024-3-13 17:44 CDT)
Removed
* cold_palette

Added
* POLA_5 palette

Changed
* Changed most of the examples to better reflect the capabilities of the program

## 0.0 - First Commits

### 0.0.5 (2024-3-10 19:48 CDT)
Changed
*  python .\main to python .\main.py

### 0.0.4 (2024-3-10 19:39 CDT)
Changed
* Fixed a \ to a / so it will display in an example added in 0.0.3

### 0.0.3 (2024-3-10 19:38 CDT)
Added
* More examples, palettes, and updated the README to show the examples

***-- went to eat at Zuma between these two versions, yellowtail fueled the next few commits***

### 0.0.2 (2024-3-10 16:29 CDT)
Changed
* Images would not display because \ in the markdown would not display in GitHub, but would in Windows with VSCode and Markdown plugin. Needed to change to /

### 0.0.1 (2024-3-10 16:26 CDT)
Added
* First examples

### 0.0.0 (2024-3-10 16:14 CDT)
First commit