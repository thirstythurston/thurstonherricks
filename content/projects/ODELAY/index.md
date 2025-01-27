---
title: "ODELAY!"
description: "One-Cell Doubling Evaluation of Living Arrays of Yeast"
tags: ["yeast", "high-content-screaning", "python"]
weight: 3

---

## Introduction


{{< youtube sz0E19Bpmx8 >}}


Growth is the most sensitive indicator of genetic fitness.  Baker's yeast is a especially attractive organism for study due to its ease of cultivation and it's genetic tractability.  However, not all cells that are genetically identical grow at the same rate; they have heterogenious growth rates.   

We developed the ODELAY! (One-Cell Doubling Evaluation of Living Arrays of Yeast) method,  using timelapse microscopy to quantify heterogenious growth of large numbers of individual yeast cells growing into colonies.  The goal of this project was to create a high-throughput growth assay that could measure this growth heterogeneity as growth heterogeneity may provide an additional dimension for characterizing genetic interactions. 

## ODELAY

{{< youtube 7iTCuV4n064 >}}

ODELAY is a timelapse microscopy method that images arrays of cells spotted onto solid agar.  Dilute cultures are spotted in arrays such that multiple strain variants can be tested at a time.  

{{< figure default=true  src="./ODELAY_overview.png  " 
    caption="ODELAY consists of a mount for the agar slide, timelapse imaging inside a microscope incubator and software to extract colony area over time."
    >}}


In this case a 8 x 12 array corresponding to the inner 96 wells of a 384 well plate were spotted onto a agar pad and imaged over 48 hours.

{{< youtube xfTUBwW_FhE >}}

This ODELAY method was considered moderate to high-throughput.  We later used it to better understand augment cryo-electron microscopy and X-ray crystal structures of the nuclear pore complex.  We did this by overlaying growth data of yeast strains with truncated protiens that made up the nuclear pore complex (nups).  

{{< figure 
    default=true  
    src="Nups_overview.png" 
    caption="ODELAY helped identify critical structural elements of the [nuclear pore complex](https://doi.org/10.1038/nature26003).  By measuring the growth rates of truncation mutants, we found that the severity of the growth defect corresponded to the importance of the truncated protein element. This technique enabled us to determine how different protein elements contribute to the structural stability of the nuclear pore complex."
    >}}

## ODELAM

We modified the ODELAY method and addapted the method for measuring growth of *Mycobacterium tuberculosis* (ODELAM), the organism that causes the respiratory disease tuberculosis.  Developing ODELAM required developing protocals and equipment that were compatible with working in a Biohazard Level 3 environment.   


{{< figure 
    default=true  
    src="ODELAM_process.png" 
    caption=""
    >}}

ODELAM required designing and fabricating a new growth chamber and techniques compatible with working entierly within a biological safety cabinet and was composed of materials that were compatible with decontamination solvents.  This included developing a tool to aide in spotting *Mycobacterium* cultures onto agar pads and a hermetically sealed growth chamber that safely contained the bacterium. 

{{< figure 
    default=true  
    src="ODELAM_output_example.png" 
    caption=""
    >}}

Above is an example of the output from a ODELAM experiment depicting 80 spotted cultures across 5 conditions. The histogram plots depict key growth parameters, including doubling time, lag time, duration of exponential growh phase, the number of observed doublings, and the population size of each spot.   This data allowed us to identify subpopulations of drug-resistant *Mycobacterium tuberculosis* (Mtb) in patient-isolated samples.