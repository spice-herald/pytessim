import numpy as np
import argparse
from detprocess import YamlConfig, Salting, RawData
from pytessim.utils import YamlTopConfig
from pytessim import Background_Manager
from utils import string_to_time, time_to_string, gen_metadata
import datetime as dt
from qetpy.utils import convert_channel_name_to_list,convert_channel_list_to_name
from detprocess.utils import create_series_name
from pytesdaq.io import H5Writer, H5Reader
from pytessim import Noise_Factory
import gc
import os
import vaex as vx
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch Raw Data Generation')

    parser.add_argument('--time',
                        dest = 'time', type=int, required=True,
                        help='Total data quantity (in seconds) to generate')
    
    parser.add_argument('--nchannels',
                        dest = 'nchannels',type=int, required=True,
                        help='Total Number of Channels to be simulated')
    
    parser.add_argument('--date-time-string','--date_time_string',
                        dest = 'date_time_string', type = str, required = False)
    
    parser.add_argument('--enable-noise', '--enable_noise',
                        dest = 'enable_noise', action='store_true',
                        help='Whether to add simulated noise to data. Default no')
    
    parser.add_argument('--enable-LEE', '--enable_LEE',
                        dest = 'enable_LEE', action='store_true',
                        help='Whether to add simulated LEE to the data. Default no')
    
    parser.add_argument('--enable-background', '--enable_background',
                        dest = 'enable_background', action='store_true',
                        help='Whether to add simulated LEE to the data. Default no')
    
    parser.add_argument('--filesize-lim-GB', '--filesize_lim_GB',
                        dest='filesize_lim_GB', type = float, required = False,
                        help='Limit of a given file size in GB')
    
    parser.add_argument('--event-length-sec', '--event_length_sec',
                        dest='event_length_sec', type = float, required = False,
                        )
    parser.add_argument('--series-length-sec', '--series_length_sec',
                        dest='series_length_sec', type = float, required = True)

    parser.add_argument('--processing-setup', '--processing_setup',
                        dest = 'processing_setup',type=str, required = True,
                        help='Processing setup (yaml) file path')  
    
    parser.add_argument('--topology-setup', '--topology_setup',
                        dest = 'processing_setup', type='str', requred = False,
                        help='Topology setup (yaml) file path')
    
    parser.add_argument('--template-file', '--template_file',
                        dest = 'template_file',type=str, required = True,
                        help='Template (.hdf5) file path')  
    parser.add_argument('--save-path', '--save_path',
                        dest='save_path', type = str, required = True)
    
    parser.add_argument('--comment', '-c',
                         dest='comment', type = str, required = True)





    #####################################
    # Parsing Arguments/Precalculations #
    #####################################

    args = parser.parse_args()

    enable_noise = False
    enable_LEE = False
    enable_background = False

    if args.enable_LEE:
        enable_LEE = True
    if args.enable_noise:
        enable_noise = True
    if args.enable_background:
        enable_background = True

    if not enable_LEE and not enable_noise:
        raise ValueError('Generating neither noise nor LEE. At least one action is required')
    
    filesize_lim_GB = 1
    if args.filesize_lim_GB:
        filesize_lim_GB = args.filesize_lim_GB

    time_sec = args.time

    n_channels = args.nchannels

    date_time_string = time_to_string( dt.datetime.now() )
    if args.date_time_string:
        date_time_string = args.date_time_string


    event_length_sec = 10 #10 events series by default
    if args.event_length_sec:
        event_length_sec = args.event_length_sec

    series_length_sec = 600 #10 minute series by default
    if args.series_length_sec:
        series_length_sec = args.series_length_sec

    save_path = args.save_path

    if save_path[-1] != '/':
        save_path +='/'
    save_path += 'continuous_I9_'+date_time_string 
    

    n_series = int(np.ceil(time_sec/series_length_sec))

    if series_length_sec % event_length_sec != 0:
        raise ValueError(f'Event length ({event_length_sec} seconds) does not '+
                         'evenly divide the series length ({series_length_sec} +'
                         'seconds).')
    
    n_events_per_series = int(series_length_sec /event_length_sec)

    comment = ''
    if args.comment:
        comment = args.comment

    processing_file_path = args.processing_setup

    if (args.topology_setup is None) != (not enable_background ):
        raise ValueError('Must include topology setup file if you want to simulate bakcgrounds')
    
    elif enable_background:
        topology_file_path = args.topology_setup


    template_file_path = args.template_file
    channel_list = []
    for i in range(n_channels):
        channel_list.append('channel_'+str(i).zfill(2))
    config = YamlConfig(processing_file_path, channel_list)

    channel_index_dict = {}
    for i in range(n_channels):
        channel_index_dict['channel_'+str(i).zfill(2)] = i

 
    ##################################
    # Setting up the Noise Factories #
    ##################################
    
    if enable_noise:
        
        noise_factory = Noise_Factory(template_file_path, event_length_sec = event_length_sec, fs = 1.25e6)

        noise_config = config.get_config('noise')

        for chan, chan_config in noise_config['channels'].items():

            # display
            print(f'INFO: Generating noise factory for channel  {chan}')

            # check if multi-channel
            chan_list = convert_channel_name_to_list(chan)
            
            
            # get config
            csd_tag = chan_config['csd_tag']

            if len(chan_list) > 2:
                raise ValueError('ERROR: reading .yaml file noise section; channel {chan} '+
                                 'is a CSD for more than 2 sensors, which is unsupported ')
            noise_factory.initialize_factory(chan)


    ##############################
    # Writing Files/Adding Noise #
    ##############################
    print('# of events per series: ' + str(n_events_per_series))

    #Increment 1 second to emulate the setup delay for data-taking
    #really just ensuring that the data folder doesn't contain the first series string
    start_time = string_to_time(date_time_string) + dt.timedelta(seconds = 1)
    start_epoch = start_time.timestamp()

    time_counter = 0
    writer = H5Writer()
    while time_counter < time_sec: #continue loop until we've filled the alotted amount of time
        n_event_counter = 0
        series_string = time_to_string(start_time + dt.timedelta(seconds = time_counter))
        writer.initialize('I9_'+series_string, data_path = save_path)
        file_metadata, adc_metadata, det_metadata = gen_metadata(n_channels, fs = 1.25e6, sec_per_event = event_length_sec,
                                                                 sec_per_series = series_length_sec, 
                                                                 dataset_start_epoch = start_epoch, 
                                                                 series_string = series_string)
        

        file_metadata['timestamp'] = start_epoch + time_counter
        file_metadata['series_start'] = start_epoch + time_counter
        writer.set_metadata(file_metadata = file_metadata, adc_config = adc_metadata, detector_config = det_metadata)
        writer._open_file(prefix = 'cont')
        
        #Add events until the series is full
        while n_event_counter < n_events_per_series: 
            
            #Check on the filesize
            current_filename = writer._current_file_name
            size_bytes = os.path.getsize(current_filename)

            #If the current file is full, close it and open another
            if size_bytes/1e9 > filesize_lim_GB:
                print('Filled Fill; starting new dump')
                writer._close_file()

                #Update timestamp and dump number parameters in metadata:
                file_metadata['timestamp'] = start_epoch + time_counter
                file_metadata['dump_number'] += 1
                writer.set_metadata(file_metadata = file_metadata, adc_config = adc_metadata, detector_config = det_metadata)
                writer._open_file(prefix = 'cont')


            if enable_noise:

                data = np.zeros((n_channels, int(1.25e6 * event_length_sec)))

                #loop over the noise templates specified in the .yaml file
                for chan in noise_factory.channel_list:

                    #Generate data for a given noise template
                    temp_data = noise_factory.sample_noise(chan)

                    #Place the data in its correct row(s) in the array
                    constituent_channels = convert_channel_name_to_list(chan)

                    if len(constituent_channels) > 1:
                        for i, name in enumerate(constituent_channels):
                            data[channel_index_dict[name]] = temp_data[i]
                    elif len(constituent_channels) == 1:
                        for name in constituent_channels:
                            data[channel_index_dict[name]] = temp_data
               
            # If not generating noise, just leave zero
            else:
                data = np.zeros((n_channels, int(1.25e6 * event_length_sec)))

            #Convert to ADC with hard-coded values stored in the metadata
            data *= 1e6 #Convert from TES current to SQUID voltage (close loop norm currently hard-coded)
            data /= 10 #Convert to [-.5,+.5]
            data *= 2**16

            #Convert to int16 (true to the actual saved format)
            #Clip to circumvent integer overflow
            data = np.clip(data, -32768, 32767).astype(np.int16)

            #Add the above data to the file
            writer.write_event(data,start_epoch + time_counter)

            #update event counter
            n_event_counter += 1
            time_counter += event_length_sec

        #Close file and update the time counter
        writer._close_file()
    
    if enable_noise:
        print('INFO: Finished file generation with noise.')
    else:
        print('INFO: Finished file generation without noise.')

    #########################################
    # Setting up the Salting class for adding LEE #
    #########################################

    if enable_LEE:
        #We're using the salting framework to add LEE pulses

        LEE_config = config.get_config('salting')

        LEE_factory = Salting(template_file_path) 

        rawdata_obj = RawData(save_path,
                              data_type='cont')


        LEE_factory.set_raw_data(rawdata_obj)

        if 'nsalt' in LEE_config['overall']:
            print('Warning: Ignoring argument nsalt; LEE counts should'
                  ' be defined as a rate for each channel.')
        
        if 'energies' in LEE_config['overall']:
            print('Warning: Ignoring argument energies; LEE distributions'
                  ' should be defined with a pdf tag/file for each channel.')

        LEE_PDF_file = None

        if 'LEE_PDF_file' in LEE_config['overall']:
            LEE_PDF_file = LEE_config['overall']['LEE_PDF_file']
        else:
            raise ValueError('ERROR: if enable_LEE is set to true, user must'
                             ' give a path to the LEE_PDF_file in the .yaml')


        chan_dataframe_list = []
        salting_dataframe_list = []
        for chan, chan_config in LEE_config['channels'].items():

            print(f'INFO: Generating LEE for channel {chan}')

            chan_list = convert_channel_name_to_list(chan)

            template_tag = chan_config['template_tag']
            pdf_tag = chan_config['pdf_tag']
            pdf_bounds = chan_config['pdf_bounds']
            rate = chan_config['rate']
            dpdi_poles = None
            dpdi_tag = None
            if 'dpdi_tag' in chan_config:
                dpdi_tag = chan_config['dpdi_tag']
            if 'dpdi_poles' in chan_config:
                dpdi_poles = chan_config['dpdi_poles']


            if (dpdi_tag is None) != (dpdi_poles is None): 
                raise ValueError("Both 'dpdi_tag' and 'dpdi_poles' must be either set or None.")
                
            if dpdi_tag is None and dpdi_poles is None:
                print(f'INFO: dpdi_tag is {dpdi_tag} and dpdi_poles is {dpdi_poles}!'
                      ' Template amplitude is assumed to be normalized to 1 eV!')
                    
            if dpdi_tag is not None and dpdi_poles is not None:
                print(f'INFO: dpdi_tag is {dpdi_tag} and dpdi_poles is {dpdi_poles}!'
                        ' Template amplitude is assumed to be 1!')
                
            pce = 1
            if 'collection_efficiency' in chan_config:
                pce = chan_config['collection_efficiency']
            elif len(chan_list) >=2:
                pce = [pce]*len(chan_list)
         
            LEE_factory.generate_salt(chan,
                                      energies=None,
                                      template_tag=template_tag,
                                      dpdi_tag=dpdi_tag,
                                      dpdi_poles=dpdi_poles,
                                      pdf_file=LEE_PDF_file,
                                      pdf_tag=pdf_tag,
                                      pdf_bounds=pdf_bounds,
                                      PCE=pce,
                                      rate=rate,
                                      poisson=True,
                                      do_salt_deadtime = True)                            
                
            LEE_dataframe = LEE_factory.get_dataframe()
            chan_dataframe_list.append(LEE_dataframe)
            i += 1
            LEE_factory.clear_dataframe()

        # concatanate as needed
        final_dataframe = chan_dataframe_list[0]
        if len(chan_dataframe_list) > 1:
            final_dataframe = vx.concat(chan_dataframe_list)

        # save to hdf5 
        series_name = create_series_name(9)
        file_name = f'LEE_pdf_{series_name}_F0001.hdf5'

        if not os.path.exists(f'{save_path}/salting'):
            os.mkdir(f'{save_path}/salting')
            print(f'INFO: Creating Salting Directory at {save_path}/salting')

        salting_file_path = f'{save_path}/salting/{file_name}'
        final_dataframe.export_hdf5(salting_file_path, mode='w')
        final_dataframe.close()

        salting_dataframe_list.append(salting_file_path)

        del final_dataframe
            
        del LEE_factory
        gc.collect()

    ######################
    # Adding LEE to file #
    ######################
    if enable_LEE:
        LEE_factory = Salting(template_file_path)
        LEE_factory.set_dataframe(salting_dataframe_list)
        file_editor = H5Reader(edit_mode=True) 
        file_editor.set_files(save_path)
        while True:

            old_traces, metadata = file_editor.read_next_event(include_metadata=True,adctoamp=True)
            if metadata['error_msg'] == 'No more files available':
                break

            #Use salting object to inject relevant salting events 
            new_traces = LEE_factory.inject_raw_salt(channel_list, old_traces, metadata['series_num'], metadata['event_num'])

            #Convert to ADC with hard-coded values stored in the metadata
            new_traces *= 1e6 #Convert from TES current to SQUID voltage (close loop norm currently hard-coded)
            new_traces /= 10 #Convert to [-.5,+.5]
            new_traces *= 2**16
            
            #Convert to int16 (true to the actual saved format)
            #Clip to circumvent integer overflow
            new_traces = np.clip(new_traces, -32768, 32767).astype(np.int16)
            file_editor.set_this_event(new_traces, detector_chans = channel_list)
        file_editor.close()
    #################


    ###################################
    # Adding Backgrounds to Data 
    ###################################

    if enable_background:

        #Perform all pre-calculations; generate salting dataframes
        BG_Manager = Background_Manager(topology_file_path,
                                        event_length_sec,
                                        series_length_sec,
                                        time_sec,
                                        1.25e6)

        BG_Manager.save_hdf5()

        file_editor = H5Reader(edit_mode=True) 
        file_editor.set_files(save_path)

        while True:
            old_traces, metadata = file_editor.read_next_event(include_metadata=True,adctoamp=True)
            if metadata['error_msg'] == 'No more files available':
                break

            #Use salting object to inject relevant salting events 
            new_traces = BG_Manager.inject_raw_salt(channel_list, old_traces, metadata['series_num'], metadata['event_num'])

            #Need to build this ZZZ
            new_traces = BG_Manager.inject_simulated_waveforms('')

            #Convert to ADC with hard-coded values stored in the metadata
            new_traces *= 1e6 #Convert from TES current to SQUID voltage (close loop norm currently hard-coded)
            new_traces /= 10 #Convert to [-.5,+.5]
            new_traces *= 2**16
            
            #Convert to int16 (true to the actual saved format)
            #Clip to circumvent integer overflow
            new_traces = np.clip(new_traces, -32768, 32767).astype(np.int16)
            file_editor.set_this_event(new_traces, detector_chans = channel_list)
        file_editor.close()     




