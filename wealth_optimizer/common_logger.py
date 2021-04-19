import logging

GLOBAL_LOG_LEVEL = 10

__all__ = ['logger']
logger = logging.getLogger('log')
logger.setLevel(GLOBAL_LOG_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(GLOBAL_LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(lineno)d: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
