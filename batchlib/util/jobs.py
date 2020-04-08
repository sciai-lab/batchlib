import os
import time


def files_to_jobs(n_jobs, *file_lists):
    n_files = len(file_lists[0])
    assert all(len(fl) == n_files for fl in file_lists)

    job_file_lists = []

    for job_id in range(n_jobs):
        this_job_list = []
        for file_list in file_lists:
            this_job_list.extend(file_list[job_id::n_jobs])
        job_file_lists.append(this_job_list)

    return job_file_lists


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
