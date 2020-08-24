import os
import os.path as path
from shutil import copy, copytree


def create_project():
    curr_dir = path.abspath(path.dirname(__file__))
    template_dir = os.path.join(curr_dir, "project_template")
    file_list = os.listdir(template_dir)

    for file in file_list:
        if file == "__init__.py":
            continue
        src = path.join(template_dir, file)
        dst = file

        print(dst)
        if os.path.isdir(src):
            copytree(src, dst)
        else:
            copy(src, dst)

    try:
        os.remove('__init__.py')
    except:
        pass
    try:
        os.remove('data/__init__.py')
    except:
        pass

