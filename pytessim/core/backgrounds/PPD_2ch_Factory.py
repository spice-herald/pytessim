from pytessim.utils import YamlTopConfig
import copy
from qetpy.utils import convert_channel_name_to_list
import qetpy as qp
import numpy as np

from detprocess import FilterData

from pytessim.core.backgrounds.Background_Factory import Background_Factory


class PPD_2ch_Factory(Background_Factory):
    def __init__(self, topology_dict, filter_file):
        
        super().__init__(topology_dict, filter_file)
        

        self._left_channel = self._topology_dict['left_channel']['channel_name']
        
        self._right_channel = self._topology_dict['right_channel']['channel_name']

    def update_dataframe(self, recoil_energy, seriesID, eventID, xpos, ypos, zpos, detectorID, eventIndex, dataframe, dataframe_He):

        left_channel_energy = recoil_energy
        right_channel_energy = recoil_energy

        left_channel_amplitude = left_channel_energy / self._energy_normalization_dict[self._left_channel] * self._PCE_dict[self._left_channel]
        right_channel_amplitude = right_channel_energy / self._energy_normalization_dict[self._right_channel] * self._PCE_dict[self._right_channel]


        n = len(dataframe['index']) 

        #update left channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index',
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._left_channel}',
                        f'salt_amplitude_{self._left_channel}', 'index']

        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._left_channel])
        dataframe['salt_recoil_energy_eV'].append(left_channel_energy)
        dataframe['saltchanname'].append(self._left_channel)
        dataframe['salting_type'].append('PPD Recoil')
        dataframe[f'salt_energy_eV_{self._left_channel}'].append(left_channel_energy)
        dataframe[f'salt_amplitude_{self._left_channel}'].append(left_channel_amplitude)
        # dataframe['detectorID'].append(detectorID)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)
        
        n += 1

        #update right channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._right_channel}',
                        f'salt_amplitude_{self._right_channel}', 'index']

        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._right_channel])
        dataframe['salt_recoil_energy_eV'].append(right_channel_energy)
        dataframe['saltchanname'].append(self._right_channel)
        dataframe['salting_type'].append('PPD Recoil')
        dataframe[f'salt_energy_eV_{self._right_channel}'].append(right_channel_energy)
        dataframe[f'salt_amplitude_{self._right_channel}'].append(right_channel_amplitude)
        # dataframe['detectorID'].append(detectorID)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)

        return dataframe, dataframe_He
