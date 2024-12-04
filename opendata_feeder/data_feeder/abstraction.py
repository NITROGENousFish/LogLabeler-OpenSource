from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingInterface(ABC):
    @abstractmethod
    def embedding_list(self,list_of_logstr:List[str]) -> List[np.ndarray]:
        ...

    @abstractmethod
    def embedding_one(self,logstr:str)-> np.ndarray:
        ...
