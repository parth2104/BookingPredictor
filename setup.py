from setuptools import find_packages,setup
from typing import List

hypen_e_dot="-e ."

def get_requirements(file_path:str)->List[str]:

    requiremetns=[]
    with open(file_path) as file_obj:
        requiremetns=file_obj.readlines()
        requiremetns=[req.replace("/n","") for req in requiremetns]

        if hypen_e_dot in requiremetns:
            requiremetns.remove(hypen_e_dot)
        return requiremetns

setup(

    name='Hotel_Reservation_Prediction',
    version='0.0.1',
    author='P@rth',
    author_email='p@rth*****@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)   

 