from pytessim.utils import YamlTopConfig
import copy
from qetpy.utils import convert_channel_name_to_list
import qetpy as qp
import numpy as np

from detprocess import FilterData


class Background_Factory():
    """
    Base class for handling conversion of energy depositions in Geant4 to 
    waveforms that can be injected into raw data. This is largely based on 
    salting class from detprocess, with two exceptions:

        1) We're "salting" in backgrounds from Geant4 that have been 
        stored in a root file, so a new pipeline for reading this data in
        is required

        2) We now need to simulate detector response 

    Paremeters
    ----------
    root_file : string
        path to root file containing lists of energy depositions
    topology_config : string

    What does this thing need to be able to do:

    """

    def __init__(self, topology_dict, filter_file):

        self._topology_dict = topology_dict

        self._filter_data = FilterData()
        self._filter_data.load_hdf5(filter_file)

        self._energy_normalization_dict = {}
        self._PCE_dict = {}
        self._template_tag_dict = {}

        self._initialize()


    def _initialize(self):
        for detector, item in copy.deepcopy(self._topology_dict).items():
            # skip the required non-detector fields
            if detector == 'name' or detector == 'target_type':
                continue
            else:
                dpdi_tag = item['dpdi_tag']
                dpdi_poles = item['dpdi_npoles']
                channel_name = item['channel_name']
                template_tag = item['template_tag']
                templates, time_array, metadata = self._filter_data.get_template(item['channel_name'], tag = template_tag, return_metadata = True)  

                pretrig_samps = metadata['']  
                collection_efficiencies = item['collection_efficiency']  

                channels = convert_channel_name_to_list(channel_name)

                self._template_tag_dict[channel_name] = item['template_tag']

                #ZZZ I think this will work for both 1x1 and 2x1 templates; there's a chance it won't for the former
                for num, chan in enumerate(channels):
                    dpdi, _ = self._filter_data.get_dpdi(channel = chan, poles = dpdi_poles, tag = dpdi_tag)
                    if templates.ndim > 1:
                        self._energy_normalization_dict[chan] = qp.get_energy_normalization(time_array, templates[num], dpdi = dpdi, lgc_ev=True)
                    else:
                        self._energy_normalization_dict[chan] = qp.get_energy_normalization(time_array, templates, dpdi = dpdi, lgc_ev=True)

                    self._PCE_dict[chan] = collection_efficiencies[num]
                