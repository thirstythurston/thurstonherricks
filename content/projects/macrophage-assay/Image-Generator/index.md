---
title: "Creating an Image-Generator"
description: "Multiprocessing to serve training data to the model "
date: "2024-11-27"
lastmod: "2024-11-27"
weight: 3

tags: []
series: ["Live-Dead Assay"]
series_order: 3
---

## Section Overview

The goal here is to understand how data is organized and delivered to a model for training. Pytorch and Tensorflow have a lot of good classes and functions to deliver data to neural net models. However, depending on how the data is organized and created, occasionally we can run into data formats that are a bit different from the methods they've designed for so it's good to know how to flexably deal with different situations that may arrise.  

Imagenet images and other datasets are usually RGB images stored as mxnx3 or mxnx4 with 8 bit unsigned integers for each color channel. The images in the datasets for this project are roughly 3600 x 2700 pixels and 16 bit unsigned integer arrays which is common when dealing with microscopy and scientific imaging but not usually encounterd in consumer devices. The images need to be broken up into smaller parts for feeding into the models.  The solution presented here again my not be the most elegant but it is a working solution that hopefully uncoverse some of the mechanics which can be hiddenwhile using the more automated solutions.  


## What is the training loop.

Pytorch has a really good description of their [DataSets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).  We will attempt to replicate the important parts of these classes in a bit of custom code.  The reason I am not directly using the DataSets and DataLoaders is that understanding how they work is important to using them effectively and modifying them when using datasets that have different properites and indexing.  

The key concept in this section are python's iterables and iterators.  These are the objects that together produce new variables while a for loop executes.  The basic idea is that a iterator when called produces a iterable object.  The iteralable object is persistant and can maintain variables to keep track the data the object has produced when the  __getitem__()  method that is called by the python for loop.  A simple training loop is shown as follows.

```python
# Create a data generator
data_generator = DataGenerator(input_variables)

# The data generator produces data and labels at each iteration of the loop.
for data, labels in data_generator:

    optimizer.zero_grad()
    outputs = model(cuda_data)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

```

The DataGenerator class will appear like this:

```python
class DataGenerator():
     def __init__(self, dataPaths, batch_size):

     def __iter__(self):
        return self

     def __len__(self):
        return num_batches
    
     def __next__(self):
        
        if self.batches_sent >= self.total_batches:
            raise StopIteration

        data_batch, label_batch = read_file_at_index(index)

        self.batches_sent +=1

        return data_batch, label_batch


```
This example has the data pre-processed.  


Setting up multiprocessing stratagies
Key concepts: python iterables and iterators, multiprocessing, Queue, Pipes, Shared memory and pinned memory

The training loop:
    each iteration of the loop creates a batch of data

In this case the data requires stitching or assembly and processing to create the true masks.  

## How to do multiprocessing.
general multiprocessing

## Alternative but unreliable complexity
main processes iterable loop the __next__() request
provide processes with data to process
Check coms if data is available 
retrive multibatches 
child process loop performs a series of tasks:
check pipe for requests/reset/kills, process requested data, deliver requested data, wait for new request

## Using Queues with multiple processes
show how they work
show how they are slow with big datasets

## Shared memory:
each process has a memory space that it writes too.  No other processes have access to it
child process communicates it has written to memory:  (a implicit lock)
main process reads the memory only after recieving the data is available
main process generates batches of data and then waits on next image to become available.  

## Making sure it all closes gracefully
Processes need to ways to shut down 
exceptions on pipe polling should initiate a shutdown processes if the communication pipes break
Queues need to be emptied
all shared memory needs to be shut down unlinked and closed eliminated
Now we have a slow method for processing and serving data to the training loop "on the fly" 


    











