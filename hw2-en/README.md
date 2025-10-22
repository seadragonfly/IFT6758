# Université de Montréal

# IFT-6758B Data Science 

# Homework 2

Grading breakdown:

| Section                       | Note |
|-------------------------------|:-----:|
| Question 1 (.py)                    | 22.5  |
| Question 1 (notebook)                | 15  |
| Question 2 (.py)                    | 20  |
| Question 3 (.py)                   | 27.5    |
| Question 4 (notebook)                      | 15   |

In general, your assignments will be graded automatically, which means you must **not** modify the signature of the defined functions (same inputs and outputs).

## Context 

The goal of this assignment is to gain experience in creating a dataset from online sources, cleaning it, and performing basic visualizations. 
To do this, you will use scraping libraries, regex, and `matplotlib`.

More specifically, we will work with an audio dataset provided by Google Research described [here](https://research.google.com/audioset//download.html).
However, we will assume that it is provided in a somewhat raw format: a CSV file with YouTube identifiers, timestamps, and labels. 

It will be your job to take this CSV, download the associated audio, format it to include only the relevant segment, and clean the data so it can be used for a downstream ML tasks (for example, training an audio classifier or a generative model to create similar audio samples).

In addition, you will visualize various aspects of the dataset to better understand it. 

## Getting Started 

This assignment has 2 parts that must be completed: 

- The `.py` files contain functions that must be filled in as specified in the comments. 
- The `visualization.ipynb` notebook contains cells that must be filled in and executed. 

Start by setting up a virtual environment as you did in previous assignments and install the contents of `requirements.txt`.

It is then recommended to complete the questions in order (starting with the `.py` file and then the corresponding sections in the notebook), as the output of earlier questions is sometimes used for later ones. 

## Questions 

## 1. Understand and visualize the dataset

To begin, we want to make `audio_segments.csv` more readable and better understand the distribution of the labels.
Complete the functions in `q1.py` then fill in and run the cells in `visualization.ipynb` under the **Question 1** section.

Looking at `audio_segments.csv`, the labels of each video correspond to the label ID and not the actual label name.

## 2. Download and process the data

Now that we have cleaned the `.csv`, it is time to move on to what really interests us: the audio.
Complete the functions in `q2.py` which will be the building blocks of our small data processing pipeline.
One function should download the audio and the other should cut the audio to include only the segment mentioned in the `.csv`.

## 3. Build the dataset with a pipeline

With these building blocks, we will build a very small data pipeline to download and process the entire dataset.
Complete the functions in `q3.py` then run the cells in `visualize.ipynb`.

## 4. Visualize the audio
Now that the segments are downloaded, complete the cells in `visualize.ipynb` to listen to and visualize some of the audio samples.

# References 
- https://research.google.com/audioset//download.html
- https://regexone.com/ 