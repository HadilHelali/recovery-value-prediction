### Description of the files for the ML process

In this repository we have : 
* `model.py` : contains the different function like train , evaluate and plot 
* `train.py` : use the train function in `model.py` and save the model
* `evaluate.py` : use the evaluate function in `model.py` and save the results (metrics)
* `preprocess_data.py` : preprocess the data then save it as well as the features 
> Note : there are some `#TODO` comment to track where exactly to change the code and make use of the current 
>  one according to our model 

### Creating pipelines

We're going to build the pipelines using **DVC** using 3 components : 
* **Inputs**: `<dataset.csv>` and `preprocess.py` script
* **Outputs**: `<dataset>_processed.csv`
* **Command**: `python preprocess.py weatherAUS.csv`

#### 1. Preprocessing Stage 

```commandline
dvc run -n preprocess \
  -d ./src/preprocess_data.py -d data/<dataset>.csv \
  -o ./data/<dataset>_processed.csv \
  python3 ./src/preprocess_data.py ./data/<dataset.csv>
```

#### 2. Training Stage 

```commandline
dvc run -n train \
  -d ./src/train.py -d ./data/<dataset>_processed.csv -d ./src/model.py \
  -o ./models/model.joblib \
  python3 ./src/train.py ./data/<dataset>_processed.csv ./src/model.py 200
```
> Note : 
> * don't forget to run `dvc push`
> * The number 200 at dvc run above is related to our script function. If your are using your own script just ignore it.
> * Creating a stage will create the following files : `dvc.yaml` and `dvc.lock`
> * So if you wanna create or change a specific stage, it's possible to just edit `dvc.yaml` . 

#### 3. Evaluation Stage 

```commandline
dvc run -n evaluate -d ./src/evaluate.py -d ./data/<dataset>_processed.csv \
  -d ./src/model.py -d ./models/model.joblib \
  -M ./results/metrics.json \
  -o ./results/precision_recall_curve.png -o ./results/roc_curve.png \
  python3 ./src/evaluate.py ./data/<dataset>_processed.csv ./src/model.py ./models/model.joblib
```
>Note : run `dvc metrics show` to show the metrics 

The DVC pipeline would look like this :
```
 +-------------------------+  
 | data/weatherAUS.csv.dvc |  
 +-------------------------+  
               *              
               *              
               *              
        +------------+        
        | preprocess |        
        +------------+        
         **        **         
       **            **       
      *                **     
+-------+                *    
| train |              **     
+-------+            **       
         **        **         
           **    **           
             *  *             
         +----------+         
         | evaluate |         
         +----------+     
```

To show the difference in **the metrics** between the difference branches : 
```commandline
dvc metrics diff
```
To reproduce a stage you can add the `stage_name` else you can reproduce all the stages without 
any specification
```commandline
dvc repro <stage_name>
```