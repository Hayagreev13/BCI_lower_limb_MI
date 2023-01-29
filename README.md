# Meta Transfer Learning For Real-Time Lower Limb MI Decoding With A Low-Cost EEG Device

## Description
The field of assistive technologies is ameliorating by the introduction of brain computer interfaces (BCIs) coupled with electrophysiological signals such as Electroencephalogram (EEG) to control devices. EEG based BCIs have applications in the mobility rehabilitation domain where the most common BCIs use motor imagery (MI), defined as the cognitive process of imagining the movement of your own body part without moving said body part. Recently, MI BCI for lower limb has been explored. However, the majority of lower limb MI BCIs studies are conducted offline producing unrealistic EEG signals when compared to real-life self-initiated movement. These studies commonly use feature extraction methods such as CSP, RG and DL. In this paper, we build a high accuracy real time lower limb MI-decoding BCI using MTL with a low-cost EEG device that is easy to use and portable. This MI BCI system is developed following the comparison of three feature extraction methods (CSP, RG and DL) in a set of closed-loop and open-loop experiments.

In the end, user could play a game of avoiding asteroids with a spaceship in real-time using lower-limb motor imagery.

## Installation instructions
```
# Create a virtual environment:
conda create --name BCI --python=3.9
conda activate BCI
# Install src to run scripts in src folder:
pip install -e .
# Install required packages:
pip install -r requirements.txt

# For running experiments with psychopy with data collection using pylsl:
conda install -c conda-forge psychopy
pip install pylsl
```

## Usage instruction
All the scripts used for the experiment have been placed in different folders based on their usage.
The scripts are numbered based on their order of exceution.
Scripts can be ran by using the command line, as example for pre-processing of subject X01 for CSP pipeline:
```
python scripts/offline_openloop_exp/data_analysis/1_preprocess.py --subjects X01 --pline csp
```
The two main folders inside scripts are for the offline-openloop experiments and online-closedloop experiments.
The offline-openloop experiments folder contains scripts for data collection and data analysis.
Data collection is done with the help of Unicorn EEG headset from g.tec, along with the python libraries - psychopy and pylsl.
Data analysis is performed to understand the data and different ML algorithms such as CSP, Reimann Goemetry and Transfer Learning have been used in the process of building the meta transfer learning model.
The 'mups' folder contains the code related to Meta transfer learning for the final model.
The final online experiments are done with the help of the scripts in 'online-closedloop exp'.

This project was mainly done using google colab to utilise the power of GPUs. Hence, notebooks for all the scripts have also been provided to perform the same.

Note that for above scripts, a data folder is needed, creating by running experiments or download data.
Please check if the path is correct.

Format of data obtained after data collection (.csv)

![Example data](/example_data.jpg "Example data")

## References

1) [MUPS-EEG](https://github.com/tiehangd/MUPS-EEG)
2) [Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification](https://arxiv.org/abs/2003.06113)