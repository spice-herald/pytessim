
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

        self._channel = self._topology_dict['channel']['channel_name']
        
    def update_dataframe(self, recoil_energy, seriesID, eventID, xpos, ypos, zpos, detectorID, eventIndex, dataframe, dataframe_He):

        channel_energy = recoil_energy

        channel_amplitude = self._energy_normalization_dict[self._channel] * self._PCE_dict[self._channel]

        n = len(dataframe['index']) 

        #update left channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._channel}',
                        f'salt_amplitude_{self._channel}', 'index']

        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._channel])
        dataframe['salt_recoil_energy_eV'].append(channel_energy)
        dataframe['saltchanname'].append(self._channel)
        dataframe['salting_type'].append('PPD Recoil')
        dataframe[f'salt_energy_eV_{self._channel}'].append(channel_energy)
        dataframe[f'salt_amplitude_{self._channel}'].append(channel_amplitude)
        # dataframe['detectorID'].append(detectorID)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)

        return dataframe, dataframe_He


