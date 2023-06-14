# Data

Here goes the data for the project, such as training datasets. When using DVC, it should point at this folder

Here we used **google drive** as a remote storage

we added the dvc dependency to the `requirements.txt`

* To list the different remote storages with dvc
```commandline
dvc remote list 
```
* adding the data to the remote storage 

```commandline
git rm -r --cached 'data/<dataset.csv>' # or add the data path in .gitignore
dvc add data/<dataset.csv>
```
> Note : this would generate a file `<dataset.csv.dvc>` from which we can track the data and even restore it when we need to with the following command 
> ```commandline
> dvc checkout data/<dataset.csv.dvc>.dvc 
> ``` 

* push the data to the dvc then commit to git 

```commandline
dvc push data/<dataset.csv>
git commit -m 'data push'
```
* checkout for an experiment branch and pull the necessary data
```commandline
git checkout branch_name
dvc pull
```