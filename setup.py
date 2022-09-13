from setuptools import find_packages, setup

with open("README.md", 'r') as f: 
    long_description= f.read() 

classifiers= [
    "Development Status :: 3 - Alpha",
    'Intended Audience :: Developers',
    "Programming Language :: Python :: 3.7",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent"
]

setup(
    name= 'deepnlp-cerelab', 
    version= '1.0.2', 
    description= 'Natural language processing package based on modern deep learning methods',
    long_description= long_description, 
    long_description_content_type= 'text/markdown',
    url= 'https://github.com/hieupth/deepnlp', 
    author= 'Dat Tien Nguyen and Hieu Trung Pham',
    author_email= 'nduc0231@gmail.com', 
    maintainer='Dat Tien Nguyen',
    maintainer_email= 'nduc0231@gmail.com',
    classifiers= classifiers,
    keywords= 'deepnlp',
    packages= find_packages(), 
    install_requires= ['transformers>= 4.21.3', 'tensorflow>=2.8.2', "numpy", "gdown>=4.4.0"],
    python_requires= ">=3.7", 
)