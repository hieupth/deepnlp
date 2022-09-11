from typing import List, Dict, Any
class TokenClassificationData:
    def __init__(self, input: List[Dict[str, Any]]):
        self.__value= input
    def value(self):
        return self.__value
    
class ParserData:
    def __init__(self, input: List[Dict[str, Any]]):
        self.__value= input
    def value(self):
        return self.__value

class MultiData:
    def __init__(self, input: List[Dict[str, Any]]):
        self.__value= input
    def value(self):
        return self.__value