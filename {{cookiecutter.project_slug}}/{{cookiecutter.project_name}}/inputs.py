import logging

from {{cookiecutter.project_name}}.base import InputData, BaseTask

logger = logging.getLogger('luigi-interface')

class KaggleInputFile(InputData):
    pass
