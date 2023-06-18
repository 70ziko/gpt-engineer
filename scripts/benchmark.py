# list all folders in benchmark folder
# for each folder, run the benchmark

import os
import subprocess

from itertools import islice
from pathlib import Path

from typer import run


def main(
    n_benchmarks: int | None = None,
):
    processes = []
    files = []
    path = Path("benchmark")

    if n_benchmarks:
        folders = islice(folders, n_benchmarks)

    for folder in benchmarks:
        if os.path.isdir(folder):
            print("Running benchmark for {}".format(folder))

            log_path = folder / "log.txt"
            log_file = open(log_path, "w")
            processes.append(
                subprocess.Popen(
                    ["python", "-m", "gpt_engineer.main", folder],
                    stdout=log_file,
                    stderr=log_file,
                    bufsize=0,
                )
            )
            files.append(log_file)

            print("You can stream the log file by running: tail -f {}".format(log_path))

    for bench_folder, process, file in benchmarks:
        process.wait()
        print("process finished with code", process.returncode)
        file.close()

        print("process", bench_folder.name, "finished with code", process.returncode)
        print('Running it. Original benchmark prompt:')
        print()
        with open(bench_folder / "main_prompt") as f:
            print(f.read())
        print()

if __name__ == "__main__":
    run(main)
