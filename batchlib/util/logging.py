import logging
import os
import sys
import glob

LOGGING_FORMATTER = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')

# log level can be passed via env var LOGLEVEL during the workflow execution
LOG_LEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
numeric_level = getattr(logging, LOG_LEVEL)

handlers = {}


def get_logger(name):
    logger = logging.getLogger(name)
    # make sure to propagate messages to parent logger, so that job loggers propagate events to workflow logger
    logger.propagate = True
    logger.setLevel(numeric_level)

    # make sure that console handler is registered only once
    if not logger.hasHandlers():
        # Always log to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(LOGGING_FORMATTER)
        logger.addHandler(stream_handler)

    return logger


def _get_filename(workflow_dir, workflow_name):
    """
    Returns log file path for the workflow and rolls over any existing logs present in the workflow_dir
    """

    def _roll_fn(fn):
        parts = fn.split('.')
        if parts[-1] == 'log':
            parts.append('1')
        else:
            log_num = int(parts[-1]) + 1
            parts[-1] = str(log_num)
        return '.'.join(parts)

    filename = os.path.join(workflow_dir, workflow_name + '.log')

    # rollover existing log files if necessary
    if os.path.exists(filename):
        # start renaming from the last
        for fn in sorted(glob.glob(filename + '*'), reverse=True):
            new_fn = _roll_fn(fn)
            os.rename(fn, new_fn)

    return filename


def add_file_handler(logger, workflow_dir, workflow_name):
    filename = _get_filename(workflow_dir, workflow_name)
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(numeric_level)
    fh.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(fh)
    # register handler
    handlers[workflow_name] = fh


def remove_file_handler(logger, workflow_name):
    fh = handlers.pop(workflow_name, None)
    if fh is not None:
        logger.removeHandler(fh)
        fh.close()
