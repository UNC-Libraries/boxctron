# Annotating Data

## Installation
We are using Label Studio to annotate our training data. To install locally:
```
pip3 install label-studio
label-studio start 
```

Further instructions are available here:
https://labelstud.io/guide/install.html#Install-with-pip

## Annotation Project Setup
1. In the Label Studio web application, create a new project, it can be named something like "Color bars v1"
2. We don't need to add anything under data at this point.
3. Under the Labels tab, and select "Object detection with bounding boxes".
4. Remove the two default labels and added "color_bar" and "subject" as labels. You will want to assign the labels different colors.

## Loading data into label studio
Images should be stored locally in a common root directory.

Then to load the images into label-studio, the suggested approach is to start a file server and load the files via a list of URLs. 

To do so, follow these instructions:
https://labelstud.io/guide/tasks.html#Run-a-web-server-to-generate-URLs-to-local-files

Which involves using this script to start the file server:
https://github.com/heartexlabs/label-studio/blob/master/scripts/serve_local_files.sh

On a Mac, the following command should work:
```
/path/to/serve_local_files.sh /path/to/normalized_files/ '"*.jpg"' train.txt 8081 
```
Which produces a file named "train.txt" containing lists of URLs to files.

Then in the Label Studio UI, we select "Import", then upload "train.txt", and select "List of tasks". The file server must be running when importing and annotating the images.

## Labeling data
Now it is ready for annotating. To start this, click on the blue "Label all tasks" button at the top of the page when the project is selected.

Now it will show you the first unlabeled image. Draw bounding boxes around the subject and color bar, going to the edge of the image for each as much as possible. If there is no color bar, then no color_bar bounding box is added.

Click "submit" or "cmd+enter" to save the annotations for the current image and move to the next one.

## Exporting Labels
Once completed, the annotations need to be exported. To do this, click on "Export" and select either "json" or "json min".