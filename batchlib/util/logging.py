import logging
import os
import sys

LOGGING_FORMATTER = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
LOG_LEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
numeric_level = getattr(logging, LOG_LEVEL)

handlers = {}


def get_logger(name):
    logger = logging.getLogger(name)
    # make sure to propagate messages to parent logger, so that job loggers propagate events to workflow logger
    logger.propagate = True
    logger.setLevel(numeric_level)
    # Always log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(stream_handler)

    return logger


def add_file_handler(logger, workflow_dir, workflow_name):
    file_path = os.path.join(workflow_dir, workflow_name + '.log')
    fh = logging.FileHandler(file_path, mode='w')
    fh.setLevel(numeric_level)
    fh.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(fh)
    # register handler
    handlers[workflow_name] = fh


def remove_file_handler(logger, workflow_name):
    fh = handlers.get(workflow_name, None)
    if fh is not None:
        logger.removeHandler(fh)
        fh.close()
