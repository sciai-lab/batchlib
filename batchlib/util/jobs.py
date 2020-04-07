

# FIXME this does not do the right thing yet
def files_to_jobs(n_jobs, *file_lists):
    n_files = len(file_lists[0])
    assert all(len(fl) == n_files for fl in file_lists)

    job_file_lists = []

    for job_id in range(n_jobs):
        this_job_list = []
        for file_list in file_lists:
            this_job_list.append(file_list[job_id::n_jobs])
        job_file_lists.append(this_job_list)

    return job_file_lists
