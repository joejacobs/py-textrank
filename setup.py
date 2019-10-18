from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="textranker",
    version="0.1",
    url="https://github.com/joejacobs/py-textranker/",
    description="Python implementation of TextRank",
    maintainer="Joe Jacobs",
    maintainer_email="joe@hazardcell.com",
    license="AGPL-3.0-or-later",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={"console_scripts": ["textranker = textranker.base:main"]},
)
