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


## | contrastive name clustering:
In order to visually observe structural differences and similarities between names, they are [clustered in 3d space](src/contrastive-name-clustering/) using the following methology. 

### - method:
The last classification layer of the already trained LSTM model is being ignored and the output-embeddings of the prior layer (dim: 32, 1) get saved.
Those will be fed as new training data into another simpler [dense-model](src/contrastive-name-clustering/coord_model.py), which predicts the corresponding x, y and z values used for plotting in 3d space.

<p align="center"> 
<img src="readme_images/clustering.png">
</p>

### - loss:
For the error calculation a custom [contrastive-cosine-similarity loss](src/contrastive-name-clustering/contrastive_loss.py) is used:

After a batch of embeddings is passed through the model, the output batch $B$ holds $|B|$ different predicted coordinates.

To create a target for the loss function, a batch $B'$ is created by flipping $B$ by 180 degrees. It now holds the same coordinates but in reverse order. The loss/similarity will be calculated by taking the mean of the cosine-similarities (cosine-angle) between each pair $B_i$ and $B'_i$.

Additionally a vector $y$ is created. For each index $i$, this vector holds either $0$ or $1$.
$y_i$ is $0$, when the coordinates $B_i$ and $B'_i$ belong to the same nationality, and $1$ if they don't.

$y = (y_1\; ...\; y_{|B|})\; ,\; y_i \in \{0; 1\}$

Then the cosine-similarity $s_i$ gets caluclated. When two coordinates at index $i$ don't belong to the same nationality, the inverted value of $s_i$ has to be used. To achieve that, it simply gets subtracted from $y_i$, which will be equal to $1$. Otherwise $s_i$ would represent the difference instead of the similarity.

$s_i = \left| -y_i + \frac{B_i \cdot B'_i}{|B_i| \cdot |B'_i|}\right|$

In addition to that, $s_i$ (loss at index $i$) is multiplied by a value $beta \in ]0;1[$, which reduces the weight of that loss. The reason: When flipping the batch $B$, the chance that $B_i$ and $B'_i$ belong the same nationality is $10\%$, since there are ten different nationalities. So there should be more weight for same-nationality coordinates, because there are far less of them. Finally, the mean of all similarities has to be calculated to get the loss $l$.

$l = \frac{1}{|B|}\sum_{i=0}^{|B| - 1} y_i \cdot \beta \cdot \left| -y_i + \frac{B_i \cdot B'_i}{|B_i| \cdot |B'_i|}\right|$


### - result:

<p align="center"> 
<img src="readme_images/rotation2.gif">
</p>