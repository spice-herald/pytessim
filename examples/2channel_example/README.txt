This folder contains all of the necessary code to simulate a TESSERACT 2-channel PPD, such as published in https://arxiv.org/abs/2503.03683. This includes:
- faithful sampling of noise from CSD (therefore including correlated noise between channels) as stored in a detprocess (.hdf5) filter file 
- Injection of both uncorrelated ("singles") and correlated ("shared") low energy excess backgrounds

The filter file/PDF files (not included due to Git's size constraints) can be generated using ff_gen.ipynb