# just out of curiosity, I would like to know how many lines of code that our codebase has.

import os 

files = os.listdir(".")
lines = 0

for file in files:
    file_path = os.path.join(".", file)
    if file_path.endswith(".py"):
        f = open(file_path, "r")
        raw_str = f.read()
        line_count = len(raw_str.split("\n"))
        lines += line_count

print("lines", lines)