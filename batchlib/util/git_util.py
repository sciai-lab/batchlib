from subprocess import check_output


def get_commit_id():
    try:
        commit_id = check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').rstrip('\n')
    except Exception:
        commit_id = None
    return commit_id
