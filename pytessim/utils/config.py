import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
import yaml
import copy
from yaml.loader import SafeLoader
import re
from pytesio import convert_length_msec_to_samples
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
from detprocess.utils import utils


__all__ = [
    'YamlTopConfig'
]

class YamlTopConfig:
    """
    Class to read and manage topology yaml configuration

    Parameters
    ----------
    yaml_file : string
        Path to topology .yaml file

    """


    def __init__(self, yaml_file):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """

        # yaml file 
        self._yaml_file = yaml_file
        
        # initialize processing config
        self._processing_config = None

        # configuration types
        self._configuration_fields = ['global','topology']

        # available global parameters
        self._overall_parameters  = {
            'global': ['filter_file', 'background_file'],
            'topology': []
            }
            

    def _read_top_config(self):
        """
        Read configuration (yaml) file 
        """
        
        # load yaml file
        yaml_dict = yaml.load(open(self._yaml_file, 'r'),
                              Loader=_UniqueKeyLoader)
        
        if not yaml_dict:
            raise ValueError('ERROR: No Topology loaded'
                             'Something went wrong...')

        # case multiple files 


        # Initialize
        processing_configs = {'global':{}}
        for field in self._configuration_fields:
            processing_configs[field] = {'overall':{},
                                         'targets':{}}
            
            
        # global parameters
        for param in self._overall_parameters['global']:
            processing_configs['global'][param] = None
            if param in yaml_dict.keys():
                processing_configs['global'][param] = copy.deepcopy(
                    yaml_dict[param]
                )
                yaml_dict.pop(param)
                
        
        # let's split configuration based on the known
        # type of processing
        # for field in self._configuration_fields:

        # check if field available
        # if field not in yaml_dict.keys():
        #     continue
        
        # set to None
        field_map = {'overall': {},
                     'targets': {}}


        # let's get config dictionary
        config_dict = copy.deepcopy(yaml_dict[field])
        yaml_dict.pop('topology')
        for config, config_items in config_dict.items():

   
            field_map['targets'][config] = (
                config_items
                )
            
        # save
        processing_configs['topology'] = field_map



            
        if ('topology' not in processing_configs
            or 'targets' not in processing_configs['topology']):
            pass
        else:
        
            new_channel_config = {}
            channels = processing_configs[field]['targets']
            for chan, chan_dict in channels.items():
                # check if disable
                if ('disable' in chan_dict and chan_dict['disable']
                    or 'run' in chan_dict and not chan_dict['run']):
                    continue

                new_channel_config[chan] = copy.deepcopy(chan_dict)
        
        # save
        processing_configs[field]['targets'] = new_channel_config

        target_reindexed = {}
        

        for target_name, target_data in processing_configs['topology']['targets'].items():
            if "G4ID" not in target_data:
                raise KeyError(f"Target '{target_name}' is missing a G4ID field")

            g4id = target_data["G4ID"]
        
            if g4id in target_reindexed:
                raise ValueError(f"Duplicate G4ID '{g4id}' found in '{target_name}'")
            
            # Store all fields except G4ID itself (since it's now the key)
            target_reindexed[g4id] = {k: v for k, v in target_data.items() if k != "G4ID"}
            target_reindexed[g4id]['name'] = target_name
        

        processing_configs[field]['targets'] = target_reindexed
    
        self._processing_config = processing_configs

    def get_config(self, subheading=None):
        """
        Get config
        """

        if self._processing_config is None:
            return None

        
        config = {}
        if subheading is not None:

            if subheading not in self._configuration_fields :
                raise ValueError(f'ERROR: Configuration type '
                                 f'"{subheading}" not found!')

            config = copy.deepcopy(self._processing_config[subheading])
            
        else:
            config = copy.deepcopy(self._processing_config)

        return config


class _UniqueKeyLoader(SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None,
                'expected a mapping node, but found %s' % node.id,
                node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f'ERROR: Duplicate key "{key}" '
                                 f'found in the yaml file for same '
                                 f'channel and algorithm. '
                                 f'This is not allowed to avoid '
                                 f'unwanted configuration!')
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping