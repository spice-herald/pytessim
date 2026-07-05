
from pytessim.utils import YamlTopConfig
import copy
from qetpy.utils import convert_channel_name_to_list
import qetpy as qp
import numpy as np

from detprocess import FilterData

from pytessim.core.backgrounds.Background_Factory import Background_Factory


class PPD_1ch_Factory(Background_Factory):
    def __init__(self, topology_dict, filter_file):
        
        super().__init__(topology_dict, filter_file)

        self._channel = self._topology_dict['channel']
        
    def update_dataframe(self, recoil_energy, seriesID, eventID, eventIndex, dataframe):

        channel_energy = recoil_energy

        channel_amplitude = self._energy_normalization_dict[self._channel] * self._PCE[self._channel]

        n = len(dataframe['index']) 

        #update left channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index',
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._channel}']

        dataframe['series_number'] = seriesID
        dataframe['event_number'] = eventID
        dataframe['trigger_index'] = eventIndex
        dataframe['salt_template_tag'][n] = self._template_tag_dict[self._channel]
        dataframe['salt_recoil_energy_eV'][n] = channel_energy
        dataframe['saltchanname'][n] = self._channel
        dataframe['salting_type'][n] = 'PPD Recoil'
        dataframe[f'salt_energy_eV_{self._channel}'][n] = channel_energy
        dataframe[f'salt_amplitude_{self._channel}'][n] = channel_amplitude
        dataframe['index'][n] = n
        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key][n] = np.nan

        return dataframe


