# name ethnicity classification

## this project uses a LSTM to assign ethnicities to names

## | currently distinguishes between these 10 countries:
```json
{
    "british": 0, "else": 1, "indian": 2, "american": 3, "german": 4, 
    "polish": 5, "pakistani": 6, "italian": 7, "romanian": 8, "french": 9, "chinese": 10
}
```

## | installation:
```bash
git clone https://github.com/hollowcodes/name-ethnicity-classification.git
cd name-ethnicity-classification/
pip install -r requirements.txt
```

## | usage:
 - ### predicting one name:
    ```
    python3 predict_ethnicitiy.py -n "Gonzalo Rodriguez" (upper-/ lower-case doesn't matter)

    >> name: Gonzalo Rodriguez - predicted ethnicity: spanish
    ```

 - ### predicting multiple names and save the output
    ```
    python3 predict_ethnicity.py -c "names.csv" "predictions.csv"
    ```

    Using the ```-c/--csv``` flag, you can predict an entire lists of names (in ```names.csv```, file name changeable) simutaneously and save them to another csv (```predictions.csv```, file name changeable).

    "names.csv" has to have one column named "names", ie.:
    ```csv
    1 names
    2 John Doe
    3 Max Mustermann
    ```

    After running the command, the "predictions.csv" will look like this:
    ```csv
    1 names,ethnicities
    2 John Doe,american
    3 Max Mustermann,german
    ```

    If the output file doesn't exist, it will be created.



## | results:

 - ### highest archived accuracy: 79.2%
 - ### loss-/ accuracy-curve and confusion matrix:
<p align="center">
<img src="readme_images/history.png">
<img src="readme_images/confusion_matrix.png">
</p>


## | cluster for visual interpretation
The data from which the clusters are created are not directly the embeddings of the names, but instead the output-embeddings ```E```, which get produced by the LSTM layer of the classifier (the last two layers are being ignored). The goal is to get an insight into the feature-extraction process of the LSTM.
(The colors represent the ground truth.)

### using random-transformation:
To create clusters using this method, the output-embeddings ```E``` are each (matrix-) multiplied with the same random matrix ```R``` and (optionally) passed into the sigmoid function.
Since the result ```C``` of this multiplication must be ```3 x 1```, so it can be plotted in 3d space, ```R``` must have the dimensions ```3 x N``` where ```N``` is the length of ```E```.


<p align="center"> 
<img src="readme_images/rand_trans.png">
</p>

#### result:
<p align="center"> 
<img src="readme_images/rt_rotation.gif">
</p>

### using principal-component-analysis:
With PCA the high-dimensional outputs embeddings get projected into 3d space.

#### result:
<p align="center"> 
<img src="readme_images/pca_rotation.gif">
</p>

### conclusions:
- british and american names are very close to each other
  
    -> probable reason: they have the same language
- british and american names are in the middle of the cluster formation
  
    -> probable reason: names of those two countries appear often in other countries

- in every cluster, there are a few names which, according to the dataset, don't belong there (false positives/negatives)
  
    -> probable reason: such names belong to people whose ancestors or who themselfes have emigrated or taken another citizenship

- the three findings above are probably largely responsible for the reduction of accuracy of the model

- nationalities with a very specific name-type (like chinese) have more dense clusters and/or are more distant from the middle