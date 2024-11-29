from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements
        
setup(
      name = 'Wafer Fault Detection',
      version = '0.0.1',
      author = 'Chetan',
      author_email='chetanfern@gmail.com',
      install_requires = [], #get_requirements('requirements.txt'),
      packages = find_packages()
      )
# We use setup.py file so that - it will treat a local folder as a package and that package we can host in pypi
# SRC folder will be treated as package and this package will be installe din our local envt like pandas and numpy using setup.py
# If a folder has a file named "__init__.py" inside it, it will be treated as package 