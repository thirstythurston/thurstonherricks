---
title:  "Malaria Research"
description:  "Investigating deformability and cytoadhesion of malaria"
tags:  ["microfluidics", "machine vision", "clinical studies"]
weight: 2
---
{{< katex >}}
## Introduction

{{< carousel images="carousel/*" interval="2500" >}}


Following my work in nanotechnology, I became interested in developing microfluidic devices to study malaria pathogenesis during my graduate studies and postdoctoral research. I designed and fabricated devices to measure the surface area and volume of red blood cells as malaria parasites grew within them. These projects involved creating single-layer PDMS-glass microfluidic devices using silicon master molds.

I chose single-layer devices without on-chip valves because they were faster and more cost-effective to develop, given my limited cleanroom fabrication time and budget. Additionally, these simpler devices were more reliable due to fewer bonding points and were easier to assemble, ensuring enough devices to provid sufficient statistical power for the study. I also developed methods to produce small production runs of approximately 500 devices for a clinical study in Blantype, Malawi.


## Microfluidics and Malaria Deformability

{{< figure
    default=true
    src="Herricks_Supp_MCD.png"
    caption="The smallest cylinder or pore that a cell can fit within has a linear relationship with surface area and volume (A).  When a malaria merozoite invades a red blood cell it wraps itself in the cell's membrane reducing the surface area and increases the cell volume (B and C).  These changes in surface area and volume shift the minimum cylindrical diameter (D) which will ultimately make cells more susceptible to filtration."
    >}}


Malaria invades and develops within red blood cells (RBCs), extensively modifying the host RBC as the parasite matures. The parasite incorporates its own proteins and lipids into the red blood cell membrane, altering both the cell's volume, surface area.  These changes alter the deformability of the cell, effectivally increasing the rate that parasitized RBC's are filtered by the spleen.  

The goal of this work was to investigate how red blood cells are filtered by the spleen, where they must deform to navigate through a tortuous path.  This path can be likened to deforming a cell to fit through a cylinder of a specific diameter. By calculating the surface area and volume of a cell, we can estimate the smallest diameter of a cylinder that a cell can fit through without increasing it's surface area. This technique allows us to determine whether a filter is likely to segregate cells based on their geometry.

{{< figure
    default=true
    src="Herricks Fig 1 Schizonts and rings v7.png"
    caption="Schizont stage parasitized red blood cells have the characteristic dark hemazoin stain settle much higher in the wedges than normal red blood cells (A).  The schizonts have a larger volume and similar surface area compaired to red blood cells (B).  When the minimum cylidrical diameter is calculated schizonts are estimated to not pass through a filter with pore sizes smaller than 3 \\(\mu\\)m (C).  Ring stage parasitized red blood cells taken from a malaria patient penetrate almost as far into the wedges as normal red blood cells (D).  However, the ring stage parasites surface area and volume (E) overlaps with normal red blood cells and their minimum cylindrical diameter suggest these cells cannot be segregated by passage through a porous filter or a tortuous path (F)."
    >}}


Wedge-shaped channels measured the surface area and volume of cells trapped within the wedge by analyzing the length and depth a cell penetrated into the channel. The differences between schizonts (mature malaria-infected cells) and ring-stage cells (recently invaded red blood cells) are evident in the following images.

The project successfully developed a model to describe how malaria parasites are filtered by the spleen. However, no significant differences were observed in the geometry of red blood cells or malaria parasites based on their presentation at admission. This finding is important because it rules out red blood cell geometry as a contributing factor to the severity of malaria infection.

## Malaria Cytoadhesion to Endothelial Cells

{{< figure
    default=true
    src="Cytoadhesion_knobs.png"
    caption="As malaria parasites remodel their host red blood cell, they create these knobs that appear on the surface of the parasitized RBC.  These knobs contain protiens that are adhesive to proteins that expressed on the interior of capillaries of specific organs such as the brain or lungs or instines etc."
    >}}

Malaria remodels red blood cells and places protiens on the surface that bind to ligands expressed by endothelial cells in specific organs.  When malaria parasites bind to [endothelial protein-C receptors (EPCR)](https://doi.org/10.1073/pnas.1524294113), the deadly condition known as cerebral malaria can develop.  This was research that was performed before the parasite-EPCR interactions were known.  

{{< youtubeLite id="pCknPsBkagA" label="Malaria parasitized red blood cells rolling on CD36" autoplay=true >}}
{{< figure default=false  src="" 
    caption="Malaria infected red blood cells demonstrating cytoadhesion in a artifical capillary"
    >}}



We developed a microfluidic culture device to grow endothelial cells in microfluidic capillaries.  

{{< figure default=false  src="microfluidic-carousel/01.jpg" 
    caption=""
    >}}

The microfluidic culture system to grow primary endothelial cells is pictured above.  I used stop-cock manifolds, instead of on chip microfluidic valves as the additional instrumentation and fabrication costs couldn't be fit in a R-21 budget.  The culture system was mounted on microscope stage inserts so that the devices could be transfered from the incubator, to biological safety cabinet, to the microscope for experiments, and back to the incubator wthout altering any fluidic connections and only having a breif reduction in flow when electrical connections were removed.  A pressure sensor measured the pressure drop across the channel and was used as feedback to a peristaltic pump to control flow rates that were gentle enough to allow growth of endothelial cells in the microfluidic chambers and capilaries while under constant flow.

{{< figure default=true  src="microfluidic-carousel/02.png" 
    nozoom=false
    caption=""
    >}}

Endothelial cells could be seeded and allowed to grow to a lawn over the course of several days.

{{< figure default=true  src="microfluidic-carousel/04.png" 
    width=500px
    caption=""
    >}}

We were able to observe and quantify parasite-endothelial cell interactions.  While this was interesting,  we were unable to observe a large number of parasite-endothelial cell interactions because the total surface area we could interogate in the microfluidic capilaries was the limiting factor.  


## Malaria Cytoadhesion to Purified Ligands

{{< youtubeLite id="U0z1m6KJ-1U" label="Malaria parasitized red blood cells rolling on CD36" autoplay=true >}}

There were two issues with measuring parasite-endothelial cell interactions:
1.  rolling on endothelial cells appeared to be a rare event
2.  we didn't know that the parasite var-gene expression matched the receptors of the endothelial cells

Consequently, we couldn't observe many interactions and the interaction we could observe we were not well characterized.  I went back to trying to find a more well controlled system and started a collaboration with [Joe Smith's lab](https://scdotorgproduction2.azurewebsites.net/research/centers-programs/global-infectious-disease-research/research-areas-and-labs/smith-lab/) that had genetically well characterized parasite strains with stable gene expression.  These "It" expressed well characterized receptors and the ligands ICAM-1 and CD36 were commercially available and adsorbed readily onto glass.  I manufactured flow cells using the same techniques to make the microfluidic devices and created flow cells that could record large populations of parasites to measure specific parasite-ligand interactions.   

{{< figure default=true  src="Var_Expression_GTS_2.png" 
    width=500px
    nozoom=false
    caption=""
    >}}

I characterized the gene expression of these ItG parasites using Q-PCR and then measured their rolling velocites on ICAM-1 or CD36 adsorbed to glass in flow cells.

{{< figure default=true  src="Clones_Rolling Velocity_Plotting_v2L.png" 
    width=500px
    nozoom=false
    caption=""
    >}}

Here is a MATLAB GUI that I wrote and used to extract rolling velocites of malaria parasites imaged in the above videos.  

{{< figure default=true  src="Figure_3_Increasing_Decreasing_WSS_Exp_Frame_DI_2.png" 
    width=500px
    rel=center
    nozoom=false
    caption=""
    >}}

The model depicted above was developed from observations that as fluid flow increased through a specific range, the parasite's rolling velocities would stay the same or in somecases actually decrease.  Thsi phenomenon is attributed to the wall shear stress created by the flowing media, which presses cells against the glass.  This effect increases the surface area of the cell in contact with the glass, thereby increasing the number of ligand-receptor interactions and making the parasites "stickier".   Consequnetly, the rolling velocity of the cells decreased as the fluid flow rate increased.  

These experiments made a good biomechanical demonstration especally because shear stress where cells got flattened against the glass corresponded to the shear stresses experienced in the post capillary venuals where parasites area observed to accumulate.  Here is where deformation actually effects survival of parastes because if they were unable to deform, they would be less sticky and potentially stay in circulation and get filtered by the spleen.     

