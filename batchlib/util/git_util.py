from subprocess import check_output

from batchlib.util import get_logger

logger = get_logger('Workflow.GitUtil')


def get_commit_id():
    try:
        commit_id = check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').rstrip('\n')
    except Exception as e:
        logger.warning(f'Cannot get git commit SHA: {e}')
        commit_id = None
    return commit_id
