release:
	python3 setup.py sdist bdist_wheel
	python3 -m pip install --upgrade twine
	twine upload --repository-url  https://upload.pypi.org/legacy/ dist/*
