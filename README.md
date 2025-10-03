# beast

![GitHub](https://img.shields.io/github/license/paninski-lab/beast)
![PyPI](https://img.shields.io/pypi/v/beast-backbones)

**Be**havioral **a**nalysis via **s**elf-supervised pretraining of **t**ransformers

`beast` is a package for pretraining vision transformers on unlabeled data to provide backbones 
for downstream tasks like pose estimation, action segmentation, and neural encoding.

## Installation

### Step 1: Install ffmpeg
First, check to see if you have ffmpeg installed by typing the following in the terminal:

```commandline
ffmpeg -version
```

If not, install:

```commandline
sudo apt install ffmpeg
```

### Step 2: Create a conda environment

First, [install anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

Next, create and activate a conda environment:

```commandline
conda create --yes --name beast python=3.10
conda activate beast
```

### Step 3: Download and install
Move to your home directory (or wherever you would like to download the code) and install via Github clone or through PyPI.

For Github cloning:

```commandline
git clone https://github.com/paninski-lab/beast
cd beast
pip install -e .
```

For installation through PyPI:

```commandline
pip install beast-backbones
```

## Usage

`beast` comes with a simple command line interface. To get more information, run
```commandline
beast -h
```

### Extract frames

Extract frames from a directory of videos to train `beast` with.

```commandline
beast extract --input <video_dir> --output <output_dir> [options]
```

Type "beast extract -h" in the terminal for details on the options.

### Train a model

You will need to specify a config path; see the `configs` directory for examples.

```commandline
beast train --config <config_path> [options]
```

Type "beast train -h" in the terminal for details on the options.

### Run inference

Inference on a single video or a directory of videos: 

```commandline
beast predict --model <model_dir> --input <video_path> [options]
```

Inference on (possibly nested) directories of images: 

```commandline
beast predict --model <model_dir> --input <video_path> [options]
```

Type "beast predict -h" in the terminal for details on the options.
