import logging
import sys
import logging
from logging import Logger
from pythonjsonlogger import jsonlogger

logger = Logger(__name__, logging.INFO)
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
