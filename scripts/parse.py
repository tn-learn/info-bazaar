import os
import gzip
import tarfile
import json

papers = {}
for archive in os.listdir('../data/papers'):
    if archive.endswith('.zip'):
        with gzip.open("../data/papers/" + archive, 'rb') as gzip_file:
            # Open the Tar archive from the Gzip file
            with tarfile.open(fileobj=gzip_file, mode='r') as tar_archive:
                # Get a list of all the members (files and directories) in the Tar archive
                members = tar_archive.getmembers()

                # Iterate through the members and print the names of files
                for member in members:
                    if member.isfile():
                        print(member.name)
                        if member.name.endswith(".tex"):
                            try:
                                papers[archive.split(".zip")[0]] = tar_archive.extractfile(member).read().decode('utf-8')
                            except Exception:
                                pass

json.dump(papers, open("../data/papers.json", "w"))
