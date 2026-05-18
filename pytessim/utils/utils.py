import datetime as dt
import numpy as np

def gen_metadata(channels, comment = '', facility = 9, fridge_run = -1, 
                 fs = 1.25e6, sec_per_event = 10, sec_per_series = 600,
                 dataset_start_epoch = 0, series_string = 'D00000000_T000000'):
    
    if isinstance(channels, int):
        n_channels = channels
        channels_list = []
        for i in range(n_channels):
            channels_list.append('channel_'+str(i).zfill(2))
    elif isinstance(channels, list):
        n_channels = len(channels)
        channels_list = channels

    file_metadata = {}
    file_metadata['comment'] = comment
    file_metadata['daq_verison'] = '0.0.0' 
    file_metadata['data_purpose'] = 'Fake Data'
    file_metadata['data_type'] = 1 
    file_metadata['dump_num'] = 1 #To start with; we can iterate this manually later
    file_metadata['facility'] = facility
    file_metadata['format_version'] = '0.0.0'
    file_metadata['fridge_run'] = -1
    file_metadata['fridge_run_start'] = 0 #Dumb number to make it clear the data are fake
    file_metadata['group_comment'] = comment
    file_metadata['group_name'] = 'continuous_I'+str(facility)+'_'
    file_metadata['group_start'] = dataset_start_epoch # 
    file_metadata['prefix'] = 'cont'
    file_metadata['run_purpose'] = 'test'
    file_metadata['run_type'] = 1
    file_metadata['series_num'] = str(facility) + series_string[1:9] + series_string[-6:]
    file_metadata['series_start'] = dataset_start_epoch #To start with; we can iterate this manually later
    file_metadata['timestamp'] = dataset_start_epoch #To start with; we can iterate this manually later



    adc_metadata = {}
    adc_channel_indices = np.arange(n_channels, dtype = int)
    adc_conversion_factor = np.tile(np.array([0,10*np.exp(-16*np.log(2)),0,0]) , (n_channels,1))
    voltage_range = np.tile(np.array([-5.0,5.0]), (n_channels,1))
    for i in range(n_channels):
        adc_metadata['connection'+str(adc_channel_indices[i])] = ['detector:'+channels_list[i],
                                                                  'controller:starcryo1_'+str(adc_channel_indices[i]+1),
                                                                  'tes:'+str(adc_channel_indices[i]+1)]
        
    adc_metadata['adc_channel_indices'] = adc_channel_indices
    adc_metadata['adc_conversion_factor'] = adc_conversion_factor
    adc_metadata['adc_mode'] = 'cont'
    adc_metadata['nb_channels'] = n_channels
    adc_metadata['nb_samples'] = int(fs * sec_per_event)
    adc_metadata['nb_events'] = int(sec_per_series/sec_per_event)
    adc_metadata['sample_rate'] = fs
    adc_metadata['voltage_range'] = voltage_range

    adc_metadata = {'adc1':adc_metadata}



    det_metadata = {}
    det_metadata['adc_channel_indices'] = adc_channel_indices
    det_metadata['adc_conversion_factor'] = adc_conversion_factor
    det_metadata['adc_name'] = 'adc1'
    det_metadata['channel_list'] = adc_channel_indices
    det_metadata['channel_type'] = 'adc'
    det_metadata['close_loop_norm'] = np.full(n_channels, 1e6)
    det_metadata['feedback_resistance'] = np.full(n_channels, 1e5)
    det_metadata['output_gain'] = np.full(n_channels, 1.0)
    det_metadata['squid_turn_ratio'] = np.full(n_channels, 10.0)

    det_metadata = {'detconfig1':det_metadata}

    return file_metadata, adc_metadata, det_metadata




def string_to_time(string):
    """
    Function for Translating between datetime object and series string
    """
    if len(string) != 17:
        raise ValueError("String should be length with Format 'DYYYYMMDD_DHHMMSS'")
    
    return dt.datetime(int(string[1:5]),
                       int(string[5:7]),
                       int(string[7:9]),
                       int(string[11:13]),
                       int(string[13:15]),
                       int(string[15:17])
                       )

def time_to_string(datetime):
    """
    Function for Translating between datetime object and series string
    """
    return datetime.strftime('D%Y%m%d_T%H%M%S')