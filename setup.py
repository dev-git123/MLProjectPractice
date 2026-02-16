from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    '''
    Docstring for get_requirements
    
    :return: list of requirements
    :rtype: List[str]
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        print("here...",requirements)
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'thin',
    author_email= 'tthlaing38@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)