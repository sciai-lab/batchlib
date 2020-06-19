import logging
import os
import sys
import glob

ROOT_LOGGER_NAME = 'Workflow'

LOGGING_FORMATTER = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s',
                                      '%Y-%m-%d %H:%M:%S')

# log level can be passed via env var LOGLEVEL during the workflow execution
LOG_LEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
numeric_level = getattr(logging, LOG_LEVEL)

handlers = {}


def _get_root_logger(name):
    return logging.getLogger(name.split('.')[0])


def _check_logger_name(name):
    root_name = name.split('.')[0]
    if root_name != ROOT_LOGGER_NAME:
        logging.warning(f"Logger '{name}' is not a child of root logger '{ROOT_LOGGER_NAME}'. "
                        f"Log events will only be directed to STDERR.")


def get_logger(name):
    _check_logger_name(name)
    logger = logging.getLogger(name)
    # make sure to propagate messages to parent logger, so that job loggers propagate events to workflow logger
    logger.propagate = True
    logger.setLevel(numeric_level)

    # make sure that console handler is registered only once in the root logger
    is_root = name.split('.')[0] == name
    if is_root and not logger.hasHandlers():
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
    return fh


def remove_file_handler(logger, workflow_name):
    fh = handlers.pop(workflow_name, None)
    if fh is not None:
        logger.removeHandler(fh)
        fh.close()
    return fh


def setup_logger(enable_logging, work_dir, name):
    if enable_logging:
        logger = get_logger('Workflow')
    else:
        logger = logging.getLogger('Workflow')
        logger.addHandler(logging.NullHandler())
    # register workflow's log file
    if enable_logging:
        fh = add_file_handler(logger, work_dir, name)
        # add file handler to tensorboard logger
        logging.getLogger('tensorflow').addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger