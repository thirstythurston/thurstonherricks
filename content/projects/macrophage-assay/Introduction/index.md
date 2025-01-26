---
title: "Introduction"
description: "What is the Assay we are trying to develop?"
weight: 1

tags: []
series: ["Live-Dead Assay"]
series_order: 1
---

## Preface

This project is essentially taking the TensorFlow [Image Segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation) and converting it into Pytorch. When I started this project in 2019 I was just learning python on still hadn't really explored different neural network toolsets.  When I picked up TensorFlow again later I ran into several problems when trying to adapt this tutorial to the live-dead assay.  Specifically, functions like using class weights for sparse cross-entropy loss functions was difficult to implement.  Some updates to the data generators were not functional and when I looked at the source it looked like code from another language was copied and pasted into the method.  Several issues in the GitHub repository had been open for years and honestly it looked like the codebase was being maintained by poorly supervised interns.    The code appears to be abandoned and consequently I gave up on TensorFlow.  

However, I thought the Image Segmentation tutorial was really good and still potentially usefull so I decided to addapt the tutorial to Pytorch. 

## Macrophages and Mycobacterium Tuberculosis

The lung disease tuberculosis is caused by a type of bacteria called *Mycobacterium tuberculosis* or *Mtb* for short. This bacterium is small and shaped like a rounded cylinder about 1 micrometer (um) in diameter and perhaps 2-4 um long. 

## Mycobacterium Tuberculosis 

{{< youtubeLite id="x_aNqX8R-9M" label="Mtb Growth" autoplay=true >}}

The lungs have immune cells that try to fight foreign organisms such as Mtb that get into the lungs.  One type of these cells, alveolar macrophages, are analogous to white blood cells in the blood.  The problem with Mtb is that it has a very waxy and tough cell wall that is apparently very resistant both physically and chemically to digestion by alveolar macrophages that try to destroy Mtb in the lungs.  

{{< youtubeLite id="PvxS79E7efc" label="Mtb Growth" autoplay=true >}}

Macrophages try to eat Mtb (see above), but then the macrophages sometimes die and the Mtb continues to grow inside the dead macrophage.  However, current assays cannot tell if Mtb grows inside *living* macrophages.  The problem is that to study Mtb in the lungs, samples and tissue usually have to be "fixed" or stabilized with chemical treatments and dyed in order to be able to see the structures and identify cells in the tissue.  The process of fixing and dying the tissue kills all the cells and Mtb in the sample.   

I was approached by [Dr. Elizabeth Gold M.D.](https://www.seattlechildrens.org/research/centers-programs/global-infectious-disease-research/research-areas-and-labs/aderem-lab/lab-team/)  and [Irina Podolskaia](https://www.seattlechildrens.org/research/centers-programs/global-infectious-disease-research/research-areas-and-labs/aderem-lab/lab-team/) to create an assay so that we could observe the events of live alveolar machrophages interacting with live Mtb.  The idea was to try and observe that when a alveolar macrophage eats (phagocitizes) a Mtb bacterium, can we keep track of the Macrophage and see does it die and Mtb grows inside the dead macrophage or does Mtb grow inside living macrophages and cause them to die.  

## Macrophages and Draq7 dye

Now it turns out there is a dye, [Draq7](https://en.wikipedia.org/wiki/Fluorophore),  which when a cell is dead will diffuse into the cell's nucleus and bind to DNA.  The Draq7 and DNA complex is fluorescent and glows a nice reddish orange color when illuminated with green light.  Most fluorescent dyes that bind to DNA tend to be cytotoxic, they kill the cell.  However, Draq7 is supposed to be harmless to cells.  Well, this is true, we found that Draq7 can become cytotoxic after it is illuminated and becomes cytotoxic over time.  

We had two sets of plates, we set one up to take time lapse images and see how well our microscope set-up worked and then all the cells died after a while.  

{{< youtubeLite id="knk0SQObjVI" label="Mtb Growth" autoplay=true >}}


We had a second plate identically prepared and all those cells were fine, until we started imaging the cells in this unexposed plate and all the cells died.    

{{< youtubeLite id="2pcgBB_Otco" label="Mtb Growth" autoplay=true >}}

But when no Draq7 was present the cells also survived while being illuminated.  

{{< youtubeLite id="5unHXrgvidk" label="Mtb Growth" autoplay=true >}}

Consequently:

{{< bootstrap-table "table table-light table-striped table-bordered" >}}
|          | + Flourescent Illumination  | - Flourescent Illumination  |
|----------|-----------------------------|-----------------------------|
|+ Draq7   |      Macrophages Die        |       Macrophages Live      |
|- Draq7   |      Macrophages Live       |       Macrophages Live      |
{{< /bootstrap-table >}}


These sets of conditions set up a really good opportunity for using a AI to classify cells as alive or dead by adding Draq7 to cultured macrophage cells, record them dyeing and then use these images as a ground truth to train a Deep Convolutional Neural Net.  We could then apply that DCNN to images of macrophages cultured with Mtb and without Draq7 and use the DCNN model to identify which cells are alive and dead without the fluorescent dyes!

The following is a description of how to go about creating a macrophage DCNN live-dead assay and what information it can give us about how Alveolar Macrophages interact with *Mycobacterium tuberculosis*.  
