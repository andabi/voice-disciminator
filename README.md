# Voice Discriminator
## Environment
* python 3.5
* tensorflow-gpu 1.8.0
* tensorpack 0.8.5

## Dataset
* A set of raw waves of target speaker. label 1
* A large set of raw waves of non-target speakers. label 0
* preprocessing required
  * consistent sample rate
* augment volumns

## TODOs
* dataset for non-target
  * optimize data pipeline
* merge/refactor mudules.py
* fix: python 2 compatibility in data.py