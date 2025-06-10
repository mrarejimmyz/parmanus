import os


def fix_eof(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    # Remove all trailing newlines and add exactly one
    new_content = content.rstrip("\n") + "\n"
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)


for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            fix_eof(os.path.join(root, file))
