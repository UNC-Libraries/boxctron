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
In the longleaf environment we are now using virtualenv as well, so installation instructions should be the same except that you will need to load the python module first:
```
module add python/3.9.6
```

# Normalizing files
The `normalize.py` script can be used to normalize images. Currently this involves resizing images, ensuring they are RGB, and converting them to JPGs.

It accepts a single file or a directory containing images. If a directory is provided, then all children directories will also be crawled for images. See the help text for usage details:

```
python3 normalize.py -h
```

# Classifier
## Model training

```
python train_color_bar_classifier.py
```

## Applying classifier to images

This script makes use of a pretrained model to normalize and then classify images. The normalized files are stored under the path configured via the `output_base_path` field, recreating the original file path there to avoid collisions.

It produces a CSV document containing the original file path, the normalized file path, the predicted class (1 = color bar, 0 = no color bar), and the confidence that the image contained a color bar. Multiple runs with the same CSV report path will append to the existing file.

Example command:
```
python classify.py -c /path/to/color_bars/ml-repo-its-config/configs/classify_locos_test.json /path/to/rbc/11500_popayan/11500_0006/ /path/to/color_bars/shared/reports/results.csv
```

## Generating a Report
`create_report.py` creates HTML report from CSV output from the classifier. It requires a path to the csv prefixed with `-f`.  <br> **Optional arguments**:
* `-s` followed by the output path of the report HTML file
* `-n` with the HTTP url to replace the normalized image path
* `-x` with the substring to indicate the area up to which the images' normalized path will be replaced by the provided HTTP url
<br>**Optional Flags**:
* `-A`  Create an aggregate report that allows you to toggle between the item-level and aggregate tables
* `-O`  Open report in browser once it has been generated

```
python create_report.py -f /path/to/file.csv -s path/to/save/report.html -n https://example.com -x /shared/ -O
```


# Segmentation Model
## Model training

```
python train_color_bar_segmenter.py
```

## Applying segmentation model to images

To use a pretrained model over a set of images (directories or individual files):
```
python segmenter_predict.py -c /path/to/color_bars/ml-repo-its-config/configs/segmenter_locos_test.json /path/to/rbc/11500_popayan/11500_0006/ /path/to/color_bars/shared/reports/seg_results.csv
```

## Generating segmentation report
Produces an HTML report from a segmenter_predict CSV output. The `-d` parameter is the path to a new directory where the report will be written. If it already exists, the script will not do anything. The `-n` parameter is optional and is used to make paths to normalized images relative rather than absolute.

```
python create_seg_report.py -f /path/to/color_bars/shared/reports/seg_results.csv -d /path/to/output/reports/new_report_dir -n /path/to/make/original/paths/relative/to
```

## Cropping images
Produces cropped versions of original image files listed in a CSV file produced by segmenter_predict. The first parameter is the path to the CSV, the second is the path where the cropped images will be written to. 

The cropped images will be written at paths based on the their original path. By default, if the original path was `/path/to/originals/a/b/image.tif`, and the output path was `/path/to/output/cropped/`, then the cropped version would be written to `/path/to/output/cropped/path/to/originals/a/b/image.jpg`. 

The `-b` option can be used to make paths shorter by making them relative to some base path, so if a `-b` option is provided as `/path/to/originals`, then the cropped version would be written to `/path/to/output/cropped/a/b/image.jpg`.

The `-e` option can be provided in order to tell the script to skip over certain images. It takes a path which should point to a CSV file, where the first column is the original path to the file to skip.

```
python crop.py /path/to/color_bars/shared/reports/seg_results.csv /path/to/output/cropped/ -b /path/to/make/original/paths/relative/to
```


# Viewing training metrics
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