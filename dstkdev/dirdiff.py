from pathlib import Path
import os
import hashlib
import sys

dir_left = r"D:\ "[:-1]
dir_right = r"E:\ "[:-1]
num_bytes = 1000
output = open(r"D:\fileright.txt", "w", encoding="utf8")


def file_hashes(dir_):
    for filepath in Path(dir_).iterdir():
        try:
            if filepath.is_file():
                file_hash = hashlib.sha1()
                file_name = filepath.as_posix()
                file_hash.update(open(file_name, "rb").read(num_bytes))
                file_size = os.path.getsize(file_name)
                yield file_name, file_size, file_hash.digest()
            elif filepath.is_dir() and filepath.as_posix() not in [r"F:/$RECYCLE.BIN"]:
                for t in file_hashes(filepath.as_posix()):
                    yield t
        except Exception as e:
            pass  # print("Skipping {} due to {}".format(filepath, e))


files_left = set()
for filename, filesize, filehash in file_hashes(dir_left):
    files_left.add((filesize, filehash))

print("{} files found".format(len(files_left)))
for filename, filesize, filehash in file_hashes(dir_right):
    if (filesize, filehash) not in files_left:
        print(filename, file=output)
