import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iuextract",
    version="0.0.10",
    author="Gecchele Marcello",
    author_email="git@gecchele.dev",
    description="Extract Idea Units from strings and files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TT-CL/iuextract",
    
    project_urls={
        "Bug Tracker": "https://github.com/TT-CL/iuextract/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires = [
        'setuptools',
        'spacy>=3.0.0',
    ]
)
