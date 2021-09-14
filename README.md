# name ethnicity classification

## | installation:

- repository installation:
    ```
    git clone https://github.com/hollowcodes/name-ethnicity-classification.git
    cd name-ethnicity-classification/
    ```
- dependencies: python-3.7, pytorch, numpy, pandas, conda recommended
  
- dependency installation using conda:
    ```
    conda create -n <env-name> python=3.7
    conda activate <env-name>

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    conda install numpy
    conda install pandas
    ```

## | usage:

 - ## read this first:

    Before you start classifying, check out the different model configurations inside the folder "model_configurations/".

    There you will find different models which each classify a unique set of nationalities.

    The "README.md" in each model folder will inform you about which ethnicities it can classify, its performance and more information you should know about it.

    When using the console interface, you can specify which model you want to use.

#

 - ## classifying names in a given .csv file :

    ### example command:
    ```
    $ python predict_ethnicity.py -i "path/to/names.csv" -o "path/to/predictions.csv" -m "standard_model_22_nationalities" -d "gpu" -b 512
    ```
    
    ### flags:
    ```
    -i, --input : path to .csv containing (first and last) names; must contain one column called "names" (name freely selectable)

    -o, --output (required after -i): path to .csv in which the names along with the predictions will be stores (file will be created if it doesn't exist; name freely selectable)

    -m, --model (optional): folder name of model configuration which can be chosen from "model_configurations/" (if this flag is not set the standard model will be "22_nationalities_and_else")

    -d, --device (optional) : must be either "gpu" or "cpu" (if this flag is not set it be "gpu" if cuda support is detected)

    -b, --batchsize (optional) : specifies how many names will be processed in parallel (if this flag is not set it will try to process all names in parallel; if it crashes choose a batch-size smaller than the amount of names in your .csv file; the bigger the batchsize the faster it will classify the names)

    ```

    ### example files:
    "names.csv" has to have one column named "names" (upper-/ lower case doesn't matter):
    ```csv
    1 names,
    2 John Doe,
    3 Max Mustermann,
    ```

    After running the command, the "predictions.csv" will look like this:
    ```csv
    1 names,ethnicities
    2 John Doe,american
    3 Max Mustermann,german
    ```

#

 - ## predicting a single name:

    ### example command:
    ```
    python3 predict_ethnicitiy.py -n "Gonzalo Rodriguez"

    >> name: Gonzalo Rodriguez - predicted ethnicity: spanish
    ```

    ### flags:
    ```
    -n, --name : first and last name (upper-/ lower case doesn't matter)
    
    -m, --model (optional): folder name of model configuration which can be chosen from "model_configurations/" (specifies which ethnicities can be classified; if this flag is not set the standard model will be "22_nationalities_and_else")
    ```

#

<br><br/>
