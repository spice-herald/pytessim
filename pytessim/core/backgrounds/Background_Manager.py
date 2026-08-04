from pytessim.utils import YamlTopConfig
import pytessim
from qetpy.utils import convert_channel_name_to_list
from detprocess import Salting
import pandas as pd
import uproot
import numpy as np
import vaex as vx

from pytessim.core.backgrounds.PPD_1ch_Factory import PPD_1ch_Factory
from pytessim.core.backgrounds.PPD_2ch_Factory import PPD_2ch_Factory
from pytessim.core.backgrounds.GaAs_1ch_Factory import GaAs_1ch_Factory
from pytessim.core.backgrounds.HeRALD_Modane_Factory import HeRALD_Modane_Factory

class Background_Manager(Salting):
    """
    Base class for handling conversion of energy depositions in Geant4 to 
    waveforms that can be injected into raw data. This is largely based on 
    salting class from detprocess, with two exceptions:

        1) We're "salting" in backgrounds from Geant4 that have been 
        stored in a root file, so a new pipeline for reading this data in
        is required

        2) We now need to simulate detector response, which will yield 
        correlations between channels in a way that can't be modeled in the 
        simplistic way that singles/shared LEE can.
    
    Background_Manager seeks to handle point #1, in addition to managing a 
    number of Background_Factory instances that will handle point #2.

    Parameters
    ----------

    root_file_path : string
        path to root file containing lists of energy depositions
    topology_path : string
        path to topology config
    filter_file_path : string
        path to 

    """

    def __init__(self,topology_config_path, event_time_sec, series_time_sec, total_time_sec, fs):

        self._config = YamlTopConfig(topology_config_path)
        self._config._read_top_config()
        self._ff_path = self._config.get_config('global')['filter_file']
        self._root_file_path = self._config.get_config('global')['background_file']
        self._event_time_sec = event_time_sec
        self._series_time_sec = series_time_sec
        self._total_time_sec = total_time_sec
        self._nevents_per_series = int(series_time_sec/event_time_sec)
        super().__init__(self._ff_path)

        self._fs = fs

        self._factory_dict = {}

        self._dataframe = None

        self._dataframe_He = None

        self._initialize_background_factories()

        self._initialize_background_df()

        self._read_root_file()




    def _initialize_background_factories(self):
        """
        Pre-generating background factories for each target volume
        """

        # list of possible factories; if you add one, you must 
        factory_list_dict = { 'PPD_1ch' : PPD_1ch_Factory,
                              'PPD_2ch' : PPD_2ch_Factory,
                              'GaAs_1ch' : GaAs_1ch_Factory,
                              'HeRALD_Modane' : HeRALD_Modane_Factory}

        topology_config = self._config.get_config()['topology']['targets']

        for target_num, target_dict in topology_config.items():
            print(f'target_num : {target_num}')
            print(f'target_dict : {target_dict}')

            target_type = target_dict.pop('target_type')
            self._factory_dict[target_num] = factory_list_dict[target_type](target_dict, self._ff_path)

    

    def _initialize_background_df(self):
        """
        Initializing the dataframe to which we will convert the G4 data into. The 
        goal is to make this dataframe as salting-like as possible, to ease
        readability AND to limit the amount of new software that needs to be written.
        """

        available_channels = []

        top_config = self._config.get_config('topology')['targets']
        for _, item in top_config.items():
            if isinstance(item, dict):
                for _, item2 in item.items():
                    if isinstance(item2, dict):
                        available_channels += convert_channel_name_to_list(item2['channel_name'])


        background_df = {'salt_template_tag' : [],
                         'salt_recoil_energy_eV' : [],
                         'saltchanname' : [],
                         'salting_type' : [],
                         'series_number' : [],
                         'event_number' : [],
                         'trigger_index' : [],
                         'index' : []
                         }
        
        for chan in available_channels:
            background_df[f'salt_energy_eV_{chan}'] = []
            background_df[f'salt_amplitude_{chan}'] = []

        self._dataframe = background_df

        he_df = {'xpos' : [],
                 'ypos' : [],
                 'zpos' : [],
                 'recoil_energy' : [],
                 'recoil_type' : [],
                 'series_number' : [],
                 'event_number' : [],
                 'trigger_index' : [],
                 'G4ID' : [], # add in functionality here for writing 
                 'index' : []}
        
        self._dataframe_He = he_df 


    def _read_root_file(self):
        """
        Read in the root file containing backgrounds to populate the background dictionary
        """

        if self._dataframe is None:
            self._initialize_background_df()
        
        
        with uproot.open(self._root_file_path) as f:
            tree = f["events;1"]
            df = tree.arrays(library="pd")
        
        G4IDs =  self._config.get_config('topology')['targets'].keys()

        df = df[df.DetectorID.isin(G4IDs)]
        
        print(f'Beginning to convert {len(df[ (df.time * 1e-9 > 0) & (df.time * 1e-9 < self._total_time_sec) ])} Geant4 energy depositions into salted pulses')

        time_counter = 0
        i_series = 0
        j_event = 0

        while True:


            event_id = int(100000 * ( i_series + 1 ) + j_event)
            series_id = time_counter # ZZZ fixme

            filt_df = df[(df.time * 1e-9 > time_counter) & (df.time * 1e-9 < time_counter + self._event_time_sec)]

            for i in range(len(filt_df)):

                event_index = int(self._fs * (filt_df.iloc[i]['time'] * 1e-9  - time_counter))

                DetectorID = filt_df.iloc[i]['DetectorID']
                energy_eV = filt_df.iloc[i]['Edep'] * 1000
                xpos = filt_df.iloc[i]['x']
                ypos = filt_df.iloc[i]['y']
                zpos = filt_df.iloc[i]['z']

                # Add relevant line(s) to the background dataframes
                self._dataframe, self._dataframe_He = self._factory_dict[DetectorID].update_dataframe(energy_eV, series_id, event_id, event_index, 
                                                                                                 xpos, ypos, zpos, DetectorID,
                                                                                                 self._dataframe, self._dataframe_He)

            j_event += 1
            time_counter += self._event_time_sec

            if j_event % self._nevents_per_series == 0:
                i_series += 1
                j_event = 0

            if time_counter >= self._total_time_sec:
                break

        self._dataframe = vx.from_dict(self._dataframe)
        self._dataframe_He = vx.from_dict(self._dataframe_He)

    def inject_raw_he_data(self, trace, seriesID, eventID):

        trace_array = trace.copy()
        
        # Ensure trace_array is 2D
        if trace_array.ndim == 1:
            trace_array = trace_array.reshape(1, trace_array.shape[-1])

        filtered_df = self._dataframe[
            (self._dataframe['event_number'] == eventID) &
            (self._dataframe['series_number'] == seriesID)
        ]

        
        # Check if filtered DataFrame is empty
        if filtered_df.count() == 0:
           
            # No salting needed -> return original trace
            return trace
        
#######3

        # he_df = {'xpos' : [],
        #          'ypos' : [],
        #          'zpos' : [],
        #          'recoil_energy' : [],
        #          'recoil_type' : [],
        #          'series_number' : [],
        #          'event_number' : [],
        #          'trigger_index' : [],
        #          'index' : []}

        # Extract common data once
        common_columns = ['recoil_energy', 'recoil_type',
                          'saltchanname', 'salting_type']
        
        common_data = {}
        for col in common_columns:

            # Extract data as NumPy arrays
            data = filtered_df.evaluate(col, array_type='numpy')

            # Check if data is a masked array
            if np.ma.isMaskedArray(data):
                # Fill masked values with np.nan
                data = data.filled(None)

            common_data[col] = data
        

        # Extract salting type once (assuming it's the same for all entries)
        salting_types = common_data['salting_type']
        salting_type = salting_types[0] if len(salting_types) > 0 else None

        for idx in range(len(filtered_df)):

            recoil_energy = common_data['recoil_energy'][idx]

            recoil_type = common_data['recoil_type'][idx]

            trigger_index = common_data['trigger_index'][idx]

            xpos = common_data['xpos'][idx]

            ypos = common_data['ypos'][idx]

            zpos = common_data['zpos'][idx]

            G4ID = common_data['G4_ID'][idx]

            self._factory_dict[G4ID].update_trace(trace, recoil_energy, trigger_index, xpos, ypos, zpos, recoil_type )




        
        # # Loop over each channel
        # for idx_channel, waveform in enumerate(trace_array):
            
        #     # Get the channel name
        #     chan = channel_list[idx_channel]
                        
        #     # Initialize the new trace for this channel
        #     newtrace = waveform.copy()

        #     # Check if the amplitude column exists for this channel
        #     amplitude_column = f'salt_amplitude_{chan}'
        #     if amplitude_column not in filtered_df.get_column_names():
        #         print(f'WARNING: No channel {chan} found in salt df! '
        #               f'Assuming single channel salt and moving on!')
        #         continue

        #     # Extract amplitude data for this channel
        #     amplitude_data = filtered_df.evaluate(amplitude_column, array_type='numpy')
        #     if np.ma.isMaskedArray(amplitude_data):
        #         amplitude_data = amplitude_data.filled(np.nan)

        #     # Iterate over the indices of the filtered DataFrame
        #     for idx in range(len(filtered_df)):

        #         # check if amplitude 
        #         saltamp = amplitude_data[idx]

        #         # Check for missing or invalid amplitude
        #         if np.isnan(saltamp):
        #             continue
        #         else:
        #             saltamp = float(saltamp)
                                
        #         # get data
        #         template_tag = str(common_data['salt_template_tag'][idx])
        #         tempchan = str(common_data['saltchanname'][idx])
        #         trigger_index = int(common_data['trigger_index'][idx])
                
        #         # Retrieve the template and times
        #         template, times = self.get_template(tempchan, tag=template_tag)
        #         nb_samples = len(times)
        #         pretrigger = nb_samples//2 
        #         # Handle tempchan containing '|'
        #         if '|' in tempchan:
        #             tempchan_list = convert_channel_name_to_list(tempchan)
        #             if chan in tempchan_list:
        #                 index = tempchan_list.index(chan)
        #                 temp = template[index][0]
        #             else:
        #                 raise ValueError(f'ERROR in inject function: '
        #                                  f'{chan} not part of  salting channel {tempchan}. '
        #                                  f'Is this correct?')
        #         else:
        #             temp = template
                
        #         # Add salting pulse
        #         saltpulse = temp * saltamp
        #         simtime = int(trigger_index)
        #         L = len(saltpulse)
        #         pretrigger = L // 2  

        #         segment = saltpulse[pretrigger:]            
        #         end = min(simtime + len(segment), len(newtrace))
        #         segment = segment[: end - simtime]  # trim if necessary
        #         newtrace[simtime:end] += segment
                    
        #     newtraces.append(newtrace)

        # # Prepare output metadata
        # output_metadata = {
        #     'salting_type': salting_type,
        #     'series_number': seriesID,
        #     'event_number': eventID
        # }
        
        # output_trace = np.array(newtraces)
     
        # if include_metadata:
        #     return output_trace, output_metadata
        # else:
        #     return output_trace