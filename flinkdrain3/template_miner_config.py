import ast
import configparser
import json
import logging
from typing import Union

from flinkdrain3.masking import MaskingInstruction,MaskingInstruction_empty

logger = logging.getLogger(__name__)

# replace original regex fill with blank
class TemplateMinerConfig_ini_modified:
    def __init__(self):
        self.profiling_enabled = False
        self.profiling_report_sec = 60
        self.snapshot_interval_minutes = 5
        self.snapshot_compress_state = True
        self.drain_extra_delimiters = []
        self.drain_sim_th = 0.4
        self.drain_depth = 4
        self.drain_max_children = 100
        self.drain_max_clusters = None
        self.masking_instructions = []
        self.mask_prefix = "<"
        self.mask_suffix = ">"
        self.parameter_extraction_cache_capacity = 3000
        self.parametrize_numeric_tokens = True

    def load(self, config_filename: str):
        parser = configparser.ConfigParser()
        read_files = parser.read(config_filename)
        if len(read_files) == 0:
            logger.warning(f"config file not found: {config_filename}")

        section_profiling = 'PROFILING'
        section_snapshot = 'SNAPSHOT'
        section_drain = 'DRAIN'
        section_masking = 'MASKING'

        self.profiling_enabled = parser.getboolean(section_profiling, 'enabled',
                                                   fallback=self.profiling_enabled)
        self.profiling_report_sec = parser.getint(section_profiling, 'report_sec',
                                                  fallback=self.profiling_report_sec)

        self.snapshot_interval_minutes = parser.getint(section_snapshot, 'snapshot_interval_minutes',
                                                       fallback=self.snapshot_interval_minutes)
        self.snapshot_compress_state = parser.getboolean(section_snapshot, 'compress_state',
                                                         fallback=self.snapshot_compress_state)

        drain_extra_delimiters_str = parser.get(section_drain, 'extra_delimiters',
                                                fallback=str(self.drain_extra_delimiters))
        self.drain_extra_delimiters = ast.literal_eval(drain_extra_delimiters_str)

        self.drain_sim_th = parser.getfloat(section_drain, 'sim_th',
                                            fallback=self.drain_sim_th)
        self.drain_depth = parser.getint(section_drain, 'depth',
                                         fallback=self.drain_depth)
        self.drain_max_children = parser.getint(section_drain, 'max_children',
                                                fallback=self.drain_max_children)
        self.drain_max_clusters = parser.getint(section_drain, 'max_clusters',
                                                fallback=self.drain_max_clusters)
        self.parametrize_numeric_tokens = parser.getboolean(section_drain, 'parametrize_numeric_tokens',
                                                            fallback=self.parametrize_numeric_tokens)

        masking_instructions_str = parser.get(section_masking, 'masking',
                                              fallback=str(self.masking_instructions))
        self.mask_prefix = parser.get(section_masking, 'mask_prefix', fallback=self.mask_prefix)
        self.mask_suffix = parser.get(section_masking, 'mask_suffix', fallback=self.mask_suffix)
        self.parameter_extraction_cache_capacity = parser.get(section_masking, 'parameter_extraction_cache_capacity',
                                                              fallback=self.parameter_extraction_cache_capacity)

        masking_instructions = []
        masking_list = json.loads(masking_instructions_str)
        for mi in masking_list:
            # instruction = MaskingInstruction_empty(mi['regex_pattern'], mi['mask_with'])
            instruction = MaskingInstruction(mi['regex_pattern'], mi['mask_with'])
            masking_instructions.append(instruction)
        self.masking_instructions = masking_instructions



class TemplateMinerConfig:
    def __init__(self,json_path = None):
        def _fallback(jsonobj,list2len,default):
            if jsonobj is None:
                return default
            try:
                return jsonobj[list2len[0]][list2len[1]]
            except:
                return default
        
        confi = None
        if json_path is not None:
            with open(file=json_path, mode='r') as f:
                confi = json.load(f)
            
        self.engine = "Drain" #Backend engine for parsing :Current Drain, JaccardDrain
        
        self.drain_sim_th:float = _fallback(confi,('drain','similar_threshold'),0.4)
        self.drain_depth:int = _fallback(confi,('drain','depth'),4)
        self.drain_max_children:int = _fallback(confi,('drain','max_children'),100)
        self.drain_max_clusters:Union[int,None] = _fallback(confi,('drain','max_clusters'),None)
        if self.drain_max_clusters == -1:
            self.drain_max_clusters = None
        
        self.profiling_enabled:bool = _fallback(confi,('profiling','enabled'),False)
        self.profiling_report_sec:int = _fallback(confi,('profiling','enabled'),60)
        
        self.snapshot_enabled:bool = _fallback(confi,('persist','enabled'),False)
        self.snapshot_interval_minutes:int = _fallback(confi,('persist','snapshot_interval_minutes'),10)
        self.snapshot_compress_state:bool = _fallback(confi,('persist','snapshot_compress_state'),False)
        
        
        self.drain_extra_delimiters:list = _fallback(confi,('mask','extra_delimiters'),[])
        self.mask_prefix:str = _fallback(confi,('mask','extra_delimiters'),"<~")
        self.mask_suffix:str = _fallback(confi,('mask','extra_delimiters'),"~>")
        self.masking_instructions = []
        for i in _fallback(confi,('mask','regex_pattern_list'),[]):
            instruction = MaskingInstruction(i['regex_pattern'], i['mask_with'])
            self.masking_instructions.append(instruction)
            
        self.parameter_extraction_cache_capacity = _fallback(confi,('other','parameter_extraction_cache_capacity'),3000)
        self.parametrize_numeric_tokens = _fallback(confi,('other','parametrize_numeric_tokens'),True)
