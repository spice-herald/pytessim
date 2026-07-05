
from pytessim.utils import YamlTopConfig
import copy
from qetpy.utils import convert_channel_name_to_list
import qetpy as qp
import numpy as np

from detprocess import FilterData

from pytessim.core.backgrounds.Background_Factory import Background_Factory


class HeRALD_Modane_Factory(Background_Factory):
    def __init__(self, topology_dict, filter_file):
        
        super().__init__(topology_dict, filter_file)

        self._channel_00 = self._topology_dict['channel_00']['channel_name']
        self._channel_01 = self._topology_dict['channel_01']['channel_name']
        self._channel_02 = self._topology_dict['channel_02']['channel_name']
        self._channel_03 = self._topology_dict['channel_03']['channel_name']
        self._channel_04 = self._topology_dict['channel_04']['channel_name']
        self._channel_05 = self._topology_dict['channel_05']['channel_name']
        self._channel_06 = self._topology_dict['channel_06']['channel_name']
        self._channel_07 = self._topology_dict['channel_07']['channel_name']
        self._channel_08 = self._topology_dict['channel_08']['channel_name']
        self._channel_09 = self._topology_dict['channel_09']['channel_name']
        self._channel_10 = self._topology_dict['channel_10']['channel_name']
        self._channel_11 = self._topology_dict['channel_11']['channel_name']
        self._channel_12 = self._topology_dict['channel_12']['channel_name']
        self._channel_13 = self._topology_dict['channel_13']['channel_name']
        self._channel_14 = self._topology_dict['channel_14']['channel_name']
        self._channel_15 = self._topology_dict['channel_15']['channel_name']
        self._channel_16 = self._topology_dict['channel_16']['channel_name']
        self._channel_17 = self._topology_dict['channel_17']['channel_name']
        self._channel_18 = self._topology_dict['channel_18']['channel_name']
        self._channel_19 = self._topology_dict['channel_19']['channel_name']
        self._channel_20 = self._topology_dict['channel_20']['channel_name']
        self._channel_21 = self._topology_dict['channel_21']['channel_name']
        self._channel_22 = self._topology_dict['channel_22']['channel_name']
        self._channel_23 = self._topology_dict['channel_23']['channel_name']


        #store templates, scaled to 1 eV speed up the real-time background generation
        self._templates = []

        for chan in [self._channel_00,  self._channel_01, self._channel_02, self._channel_03,  self._channel_04, self._channel_05, 
                     self._channel_06,  self._channel_07, self._channel_08, self._channel_09,  self._channel_10, self._channel_11, 
                     self._channel_12,  self._channel_13, self._channel_14, self._channel_15,  self._channel_16, self._channel_17, 
                     self._channel_18,  self._channel_19, self._channel_20, self._channel_21,  self._channel_22, self._channel_23 ]:
            self._templates.append(self._filter_data.get_template(chan, tag = self._template_tag_dict[chan]) * self._energy_normalization_dict[chan] * self._PCE_dict[chan])

        self._templates = np.array(self._templates)

        self._templates = np.squeeze(self._templates)
        

    def update_dataframe(self, recoil_energy, seriesID, eventID, eventIndex, xpos, ypos, zpos, detectorID, dataframe, dataframe_He):

        """
        Update 
        """
        n = len(dataframe_He['index']) 

        #update left channel  
        updated_keys = ['series_number', 'event_number', 'trigger_index', 
                        'recoil_energy', 'recoil_type', 'xpos', 'ypos',
                        'zpos', 'DetectorID', 'index']

        dataframe_He['series_number'][n] = seriesID
        dataframe_He['event_number'][n] = eventID
        dataframe_He['trigger_index'][n] = eventIndex
        dataframe_He['recoil_energy'][n] = recoil_energy
        dataframe_He['recoil_type'][n] = 'ER'
        dataframe_He['xpos'][n] = xpos
        dataframe_He['ypos'][n] = ypos
        dataframe_He['zpos'][n] = zpos
        dataframe_He['detectorID'][n] = detectorID
        dataframe_He['index'][n] = n

        for key in dataframe_He.keys():
            if key not in updated_keys:
                dataframe_He[key][n] = np.nan
        
        return dataframe, dataframe_He
    
    def update_trace(self, traces, recoil_energy, trigger_index, xpos, ypos, zpos, recoil_type):
        """
        Updates a 24-channel waveform to add a He-mediated signal
        
        Parameters
        ----------
            traces : np.array
                24 x N array corresponding to raw TES current streams
            recoil_energy : float
                recoil energy in eV
            trigger_index : int 
                index (time) at which the recoil occurs
            xpos, ypos, zpos : floats
                recoil vertex in ZZZ units
            recoil_type : string
                either 'ER' or 'NR' for electron/nuclear recoils respectively
        
        Returns
        -------
            updated_traces : np.array
                24 x N arra of TES current streams
                
        """

        HeSTSignal = self.simulate(recoil_energy, trigger_index, xpos, ypos, zpos, recoil_type)
        
        for idx in range(24):

            template = self._templates[idx] #ZZZ fixme
            pretrig_samps = 1
            posttrig_samps = len(template) - pretrig_samps
    
            times_us = HeSTSignal.arrivalTimes[idx]
            energies_eV = HeSTSignal.energies[idx]


            for idx_quant in range(len(times_us)):
                index = int(1.25 * times_us[idx_quant]) + trigger_index - pretrig_samps

                if index >= 0 and index < len(traces[0]) - posttrig_samps:
                    traces[idx][index:index+len(template)] += template * energies_eV[idx_quant] 

                elif index < 0:
                    traces[idx][:len(template)+index] += template[abs(index):] * energies_eV[idx_quant]
                
                else:
                    traces[idx][index:] += template[:len(traces[0]-index)] * energies_eV[idx_quant] 
        
        return traces

    def simulate(self, recoil_energy, xpos, ypos, zpos, recoil_type):

        return object