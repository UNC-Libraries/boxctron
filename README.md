# Repository Preingest Processing

This project is intended for use performing pre-ingest processing of image files using machine learning techniques. Python 3.8-3.10 is required to run.

# Local/Library server environment
Since the library servers use virtual environments, we use virtualenv with pip in this environment.

For library server environments, we need to enable python 3:
```
scl enable rh-python38 bash
```
To checkout the project, create a virtual environment and install dependencies:
```
cd /path/to/
git clone git@github.com:UNC-Libraries/ml-repo-preingest-processing.git
cd ml-repo-preingest-procressing
# `python3 -m venv venv` on library servers
python3 -m virtualenv venv

# active the virtual env and install dependencies
source venv/bin/activate
python3 -m pip install -r requirements.txt
```
To deactivate the environment:
```
deactivate
```
Dependencies are primarily being managed by conda in the longleaf environment, so check there for how to sync dependencies back from that environment. But pip3 can also be used to updated dependencies.


# Longleaf environment
We are using conda for managing the environment on ITS computing resources. The environment is captured in environment.yml, and creates an "envs" directory within the project directory.

To recreate the environment:

```
conda remove --name ./envs --all

module add anaconda/2023.03
conda create --prefix ./envs -c pytorch -c nvidia -c conda-forge python=3.10 pillow=9.4 pytorch=2.0 torchvision=0.15 pytorch-lightning=2.0 metaflow=2.8 pytest=7.3 pytorch-cuda=11.8 tensorboard=2.12 pandas=2.0 seaborn=0.12 scikit-learn=1.2
```
To activate the environment, run tests, and deactivate it:
```
conda activate ./envs
python -m pytest src/tests/
conda deactivate
```
To sync the dependencies from conda t?o pip requirements (note, there will often be a few dependencies that are platform specific and need to be cleaned out, like mkl-*):
```
pip list --format=freeze > requirements.txt
```

# Normalizing files
The `normalize.py` script can be used to normalize images. Currently this involves resizing images, ensuring they are RGB, and converting them to JPGs.

It accepts a single file or a directory containing images. If a directory is provided, then all children directories will also be crawled for images. See the help text for usage details:

```
python3 normalize.py -h
```

# Model training

```
python train_color_bar_classifier.py
```

# Applying classifier to images

This script makes use of a pretrained model to normalize and then classify images. The normalized files are stored under the path configured via the `output_base_path` field, recreating the original file path there to avoid collisions.

It produces a CSV document containing the original file path, the normalized file path, the predicted class (1 = color bar, 0 = no color bar), and the confidence that the image contained a color bar. Multiple runs with the same CSV report path will append to the existing file.

Example command:
```
python classify.py -c /path/to/color_bars/ml-repo-its-config/configs/classify_locos_test.json /path/to/rbc/11500_popayan/11500_0006/ /path/to/color_bars/shared/reports/results.csv
```


## Viewing training metrics
Training details are logged using tensorboard. They can be viewed by running:
```
tensorboard --logdir logs/lightning_logs/version_1
```
And then going to http://localhost:6006/

# Running tests
To run the tests in your local environment (or python3 depending on the environment):
```
python -m pytest src/tests/
```