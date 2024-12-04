import os
import time

from functools import wraps
# from abc import ABC, abstractmethod

class BasePersistentDecorator(object):
    """Usage
    1. define your decorator parameters in function do_call
    """
    def _add_dict(self, func, decorator_kwargs):
        for k, v in decorator_kwargs.items():
            func.__dict__[k] = v

    def __call__(self, *_, **kwargs):                          
        return self.do_call(**kwargs)                          

    def do_call(self, *_, **decorator_kwargs):
        def wrapper(func):
            wrapper.__explained = False

            @wraps(func)                                       
            def _wrap(*args, **kwargs):
                if not wrapper.__explained:                    
                    self._add_dict(func, decorator_kwargs)
                    wrapper.__explained = True

                return self.invoke(func, *args, **kwargs)      

            self._add_dict(_wrap, decorator_kwargs)             
            _wrap = self.wrapper(_wrap)                        
            return _wrap

        return wrapper

    def wrapper(self, wrapper):
        return wrapper

    def invoke(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class ProfileChunk:
    def __init__(self,section_name) -> None:
        self.section_name = section_name
        self.exec_time = 0
        self.exec_count = 0

    def clear(self):
        self.periodic_exec_time = 0; self.periodic_exec_count = 0

    def update(self,exec_time,exec_count):
        self.exec_time += exec_time; self.exec_count += exec_count
        
    def __str__(self) -> str:
        total_exec_time_str = f"{self.exec_time:>8.2f} s"
        
        ms_per_k_samples = f"{1000000 * self.exec_time / self.exec_count: 7.2f}"
        samples_per_sec = f" ({self.exec_count / self.exec_time: 15,.2f})"
        
        return f"{self.section_name: <15}: took {total_exec_time_str}, " + \
               f"{self.exec_count: >10,} samples, " + \
               f"{ms_per_k_samples} ms / 1000 samples, " + \
               f"{samples_per_sec} hz"

class MultiProfileChunk:
    def __init__(self) -> None:
        self.section_to_stats = {}
    def update(self,section_name,exec_time,exec_count):
        section_stats = self.section_to_stats.get(section_name, None)
        if section_stats == None:
            section_stats = ProfileChunk(section_name)
            self.section_to_stats[section_name] = section_stats
        section_stats.update(exec_time,exec_count)
    def __str__(self) -> str:
        each_section = self.section_to_stats.values()
        sorted_sections = sorted(each_section, key=lambda it: it.exec_time, reverse=True)
        lines = map(lambda it: str(it), sorted_sections)
        text = os.linesep.join(lines)
        return text

class AggregateProfilerDecorator(BasePersistentDecorator):
    def __init__(self,periodic_report_interval=30,enable=True,printer=print) -> None:
        self.periodic_report_interval = periodic_report_interval
        self.tmp_last_periodic_chunk = time.time()
        self.periodic_profile_chunk = MultiProfileChunk()
        self.total_profile_chunk = MultiProfileChunk()
        self.enable = enable #* not used
        self.printer = printer

    def __call__(self, section_name):
        return self.do_call(section_name=section_name)
    
    def report_status(self):
        if self.enable==False:
            self.printer("Profiler is disabled")
        else:
            self.printer(str(self.total_profile_chunk))
        
    def report_periodic_status_and_clear(self,now:float):
        #* report profile chunk status within periodic_report_interval
        if now-self.tmp_last_periodic_chunk > self.periodic_report_interval:
            self.printer(str(self.periodic_profile_chunk))
            self.tmp_last_periodic_chunk = now
            self.periodic_profile_chunk = MultiProfileChunk()
        
    def invoke(self, func, *args, **kwargs):
        if self.enable==False:
            return func(*args, **kwargs)
        else:
            s_time = time.time()
            self.report_periodic_status_and_clear(s_time)
            
            #*** main function
            res = func(*args, **kwargs)
            
            duration = time.time() - s_time
            self.periodic_profile_chunk.update(func.section_name,exec_time=duration,exec_count=1)
            self.total_profile_chunk.update(func.section_name,exec_time=duration,exec_count=1)
            
            return res


def NullDecorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

PROFILER = AggregateProfilerDecorator(periodic_report_interval=10)
