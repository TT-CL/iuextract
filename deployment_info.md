# Deployment Information
Follow [this link](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for more details.
---
### Testing the package
Update build tools
`python3 -m pip install --upgrade build`
Build the package
`python3 -m build`
Update twine
`python3 -m pip install --upgrade twine`
Go on [Test PyPI](https://test.pypi.org/), login and create a new login token.
Initiate the package upload
`python3 -m twine upload --repository testpypi dist/*`
Use `__token__` as your username and paste the login token as your password.
The package should upload without issues.
**Testing the package**
Create a new environment
`python3 -m venv pytestenv`
Activate the environment
`source pytestenv/bin/activate`
Install your package
`python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps iuextract`
Run a python instance and try importing `iuextract`
If everything works, you can deactivate the environment and delete it.
`deactivate`
`rm -rf pytestenv`
### Uploading the package to PyPI
Initiate the package upload
`twine upload dist/*`
Input your actual credentials to upload the package on PyPI