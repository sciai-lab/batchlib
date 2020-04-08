import os
import time


def files_to_jobs(n_jobs, file_list):
    job_lists = []
    for job_id in range(n_jobs):
        job_lists.append(file_list[job_id::n_jobs])
    return job_lists


class FileLock:
    def __init__(self, path, timeout=10):
        self.path = path
        self.timeout = timeout

    def wait_for_unlock(self):
        while True:
            time.sleep(self.timeout)
            if not os.path.exists(self.path):
                break

    def __enter__(self):
        if os.path.exists(self.path):
            self.wait_for_unlock()
        with open(self.path, 'w'):
            pass

    def __exit__(self, type, value, traceback):
        os.remove(self.path)
