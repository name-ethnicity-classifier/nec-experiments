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
 - ### confusion matrix:
<p align="center"> 
<img src="readme_images/confusion_matrix.png">
</p>

 - ### loss-/ accuracy-curve:
<p align="center"> 
<img src="readme_images/history.png">
</p>


## | cluster for visual interpretation

### method:
The colors represent the ground truth.
These clusters are created by passing the output-embeddings of the LSTM layer through a randomly initialzed, not trained model.
This model consists of only one fully-connected sigmoid layer, which outputs a vector of shape (3, 1) and can be plotted in 3d space.

The reason why this works is not clear. But it seems it's due to the very ordered feature-vector created by the LSTM.

<p align="center"> 
<img src="readme_images/rotation1.gif">
</p>

### conclusions:
- british and american names are very close to each other
  
    -> probable reason: they have the same language
- british and american names are in the middle of the cluster formation
  
    -> probable reason: names of those two countries appear often in other countries

- in every cluster, there are a few names which, according to the dataset, don't belong there (false positives/negatives)
  
    -> probable reason: such names belong to people whose ancestors or who themselfes have emigrated or taken another citizenship

- the three findings above are probably largely responsible for the reduction of accuracy of the model

- nationalities with a very specific name-type (like chinese) have more dense clusters and are more distant from the middle