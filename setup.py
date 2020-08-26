from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read()

setup(
    name="train-and-deploy",
    packages=find_packages(),
    version="0.1.4",
    description="Train and Deploy is a framework to automatize the Machine Learning workflow.",
    author="Pi Esposito",
    url="https://github.com/piEsposito/tand",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8"
    ],
    package_data={
        "tand.structured_data.classification.pytorch": ["project_template/*",
                                                        "project_template/data/*",
                                                        "project_template/env_files/*",
                                                        "project_template/lib/*"],

        "tand.structured_data.classification.sklearn": ["project_template/*",
                                                        "project_template/data/*",
                                                        "project_template/env_files/*",
                                                        "project_template/lib/*"],

        "tand.structured_data.regression.pytorch": ["project_template/*",
                                                    "project_template/data/*",
                                                    "project_template/env_files/*",
                                                    "project_template/lib/*"],
    },
    include_package_files=True,
    entry_points={
        'console_scripts': [
            'tand-create-project = tand.create_project:main',
            'tand-prepare-aws-eb-deployment = tand.deployment.aws_elastic_beanstalk:main',
        ]
    }

)
