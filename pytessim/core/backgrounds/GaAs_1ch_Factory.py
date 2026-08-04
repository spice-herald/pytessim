from pytessim.utils import YamlTopConfig
import copy
from qetpy.utils import convert_channel_name_to_list
import qetpy as qp
import numpy as np

from detprocess import FilterData

from pytessim.core.backgrounds.Background_Factory import Background_Factory

class GaAs_1ch_Factory(Background_Factory):
    def __init__(self, topology_dict, filter_file):
        
        super().__init__(topology_dict, filter_file)

        self._target_channel = self._topology_dict['target_channel']['channel_name']
        self._top_channel = self._topology_dict['top_channel']['channel_name']
        self._bottom_channel = self._topology_dict['bottom_channel']['channel_name']

    def update_dataframe(self, recoil_energy, seriesID, eventID, xpos, ypos, zpos, detectorID, eventIndex, dataframe, dataframe_He):


        target_channel_energy, top_channel_energy, bottom_channel_energy = self._simulate(recoil_energy)

        target_channel_amplitude = target_channel_energy * self._energy_normalization_dict[self._target_channel] * self._PCE_dict[self._target_channel]
        top_channel_amplitude = top_channel_energy * self._energy_normalization_dict[self._top_channel] * self._PCE_dict[self._top_channel]
        bottom_channel_amplitude = bottom_channel_energy * self._energy_normalization_dict[self._bottom_channel] * self._PCE_dict[self._bottom_channel]

        n = len(dataframe['index']) 

        #update target channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._target_channel}',
                        f'salt_amplitude_{self._target_channel}', 'index']

        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._target_channel])
        dataframe['salt_recoil_energy_eV'].append(target_channel_energy)
        dataframe['saltchanname'].append(self._target_channel)
        dataframe['salting_type'].append('GaAs Recoil')
        dataframe[f'salt_energy_eV_{self._target_channel}'].append(target_channel_energy)
        dataframe[f'salt_amplitude_{self._target_channel}'].append(target_channel_amplitude)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)       
        
        n += 1

        #update top channel
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._top_channel}',
                        f'salt_amplitude_{self._top_channel}', 'index']
        
        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._top_channel])
        dataframe['salt_recoil_energy_eV'].append(top_channel_energy)
        dataframe['saltchanname'].append(self._top_channel)
        dataframe['salting_type'].append('GaAs Recoil')
        dataframe[f'salt_energy_eV_{self._top_channel}'].append(top_channel_energy)
        dataframe[f'salt_amplitude_{self._top_channel}'].append(top_channel_amplitude)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)       

        n += 1

        #update bottom channel
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'salt_template_tag', 'salt_recoil_energy_eV', 'saltchanname', 
                        'salting_type', f'salt_energy_eV_{self._bottom_channel}',
                        f'salt_amplitude_{self._bottom_channel}', 'index']

        dataframe['series_number'].append(seriesID)
        dataframe['event_number'].append(eventID)
        dataframe['trigger_index'].append(eventIndex)
        dataframe['salt_template_tag'].append(self._template_tag_dict[self._bottom_channel])
        dataframe['salt_recoil_energy_eV'].append(bottom_channel_energy)
        dataframe['saltchanname'].append(self._bottom_channel)
        dataframe['salting_type'].append('GaAs Recoil')
        dataframe[f'salt_energy_eV_{self._bottom_channel}'].append(bottom_channel_energy)
        dataframe[f'salt_amplitude_{self._bottom_channel}'].append(bottom_channel_amplitude)
        dataframe['index'].append(n)

        for key in dataframe.keys():
            if key not in updated_keys:
                dataframe[key].append(np.nan)       
        return dataframe, dataframe_He

    def _simulate(self, recoil_energy):
        """
        Function simulating the energy partition between the three sensors. Just a dummy 
        function for now; this could get infinitely more complex and even wrap to another
        simulation package that would need to be initialized alongside this class
        """

        return .3*recoil_energy, .3*recoil_energy, .3*recoil_energy