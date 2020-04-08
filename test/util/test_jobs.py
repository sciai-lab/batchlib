import unittest
from shutil import rmtree


class TestJobs(unittest.TestCase):
    def test_files_to_jobs(self):
        from batchlib.util import files_to_jobs
        n_jobs = 3
        file_list = list(range(24))

        job_lists = files_to_jobs(n_jobs, file_list)
        self.assertEqual(len(job_lists), n_jobs)


if __name__ == '__main__':
    unittest.main()
