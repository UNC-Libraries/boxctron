# Repository Preingest Processing

This project is intended for use performing pre-ingest processing of image files using machine learning techniques.

# Installation
For local development, this project is using pipenv, so that will need to be installed:

https://pipenv.pypa.io/en/latest/install/#installing-pipenv

And then the dependencies will need to be installed:
```
pipenv install
```

Commands will need to be prefixed in order to use the local environment. For example, to display which image formats are supported by pillow:
```
pipenv run python3 -m PIL
```

On remote servers, pip and virtualenv are used.

# Updating Dependencies
When updating or adding dependencies, use pipenv:
https://pipenv.pypa.io/en/latest/install/#installing-packages-for-your-project

To keep dependencies synched for remote server use, when dependencies in the project change we will need to update the requirements.txt file using:
```
pipenv requirements
```

And those tools will need to be used to install dependencies in the remote environment
```
pip3 install -r requirements.txt
```