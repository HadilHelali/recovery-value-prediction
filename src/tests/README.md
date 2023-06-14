# Tests

We will use `Pytest` which is a Python package and a very popular choice 
for regular Software testing and can be used to implement all sorts of tests. 

To run all the tests all we have to do is running this command :
```commandline
pytest
```
The tests that need to change are  :
* `test_model.py`
* `test_preprocess.py`

We've also added the `pylint` which a widely-used static code analysis tool for Python.
We added it's dependancy in the `requirement.txt` and to use it , we need to run this command : 
```commandline
pylint ./src/*.py
```
For the time being we got a score of `6.4/10`