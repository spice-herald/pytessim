In this example, we're simulating an example datastream from a 1-channel PPD. This includes noise, singles/shared LEE (differentiated by pulse shape, rate, and energy), and backgrounds from direct strikes. The following files are relevant:
- ffgen.ipynb : used to generate the filter file (containing signal/noise templates) and LEE .pkl file (containing LEE distributions)
- PPD_1ch.yaml : manages noise + LEE injection
- PPD_1ch_topo..yaml : manages background injection

Running ffgen.ipynb should create the following files: 
- 1channel_ppd.hdf5 
- LEE_distributions.pkl

In order to generate the data, the following code can be run:

python path/to/pytessim/scripts/file_generation.py --enable-noise --enable-LEE --enable-background --event-length-sec 10 --series-length-sec 60 --time 180 --processing-setup PPD_1ch.yaml --nchannels 1 --template-file 1channel_ppd.hdf5 --save-path ./generated/ --comment test --topology-setup PPD_1ch_topo.yaml
