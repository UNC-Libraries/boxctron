# Repository Preingest Processing

This project is intended for use performing pre-ingest processing of image files using machine learning techniques.

# Installation

## Local development
For local development, this project is using pipenv:

https://pipenv.pypa.io/en/latest/install/#installing-pipenv

Dependencies are installed using:
```
pipenv install
```

Commands will need to be prefixed in order to use the local environment. For example, to display which image formats are supported by pillow:
```
pipenv run python3 -m PIL
```

## Remote development
On remote servers, pip and virtualenv are used. For first time installation:

```
scl enable rh-python38 bash
cd /path/to/
# Or git pull to update
git clone git@github.com:UNC-Libraries/ml-repo-preingest-processing.git
cd ml-repo-preingest-procressing
python3 -m venv ml-repo-preingest-procressing --system-site-packages

# active the virtual env and install dependencies
cd ml-repo-preingest-procressing
source bin/activate
cd ..
pip3 install -r requirements.txt
```
A similar workflow is used for updating the code and/or updating dependencies.

To run the normalize.py command on a remote server, you will first need to activate the virtualenv and run the command:
```
scl enable rh-python38 bash
cd /path/to/ml-repo-preingest-processing
source ml-repo-preingest-procressing/bin/activate

python3 normalize.py -h
```

# Updating Dependencies
When updating or adding dependencies, use pipenv:
https://pipenv.pypa.io/en/latest/commands/#install

To keep dependencies synched for remote server use, when dependencies in the project change we will need to update the requirements.txt file using:
```
pipenv requirements > requirements.txt
```

pip and virtualenv will need to be used to install dependencies in the remote environment
```
pip3 install -r requirements.txt
```

# Normalizing files
The `normalize.py` script can be used to normalize images. Currently this involves resizing images, ensuring they are RGB, and converting them to JPGs.

It accepts a single file or a directory containing images. If a directory is provided, then all children directories will also be crawled for images. See the help text for usage details:

Locally:
```
pipenv run python3 normalize.py -h
```

Remotely:
```
python3 normalize.py -h
```

# Running tests
To run the tests in your local environment:
```
pipenv run pytest src/tests/
```