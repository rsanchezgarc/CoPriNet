#CoPriNet

CoPriNet is a Graph Neural Network trained on pairs of molecule 2D graphs and catalogue prices. CoPriNet predictions
can be used as a proxy score for compound availability.

##Installation
```
conda env create -f CoPriNet_env.yml 
conda activate CoPriNet
```

##Scoring molecules
In order to execute CoPriNet you only need to prepare a csv file with your SMILES and execute the following command
```
python -m pricePrediction.predict path/to/csvFile -o path/toResults 

E.g.

~/CoPriNet$ python -m pricePrediction.predict data/testData/npnp_dataset.csv -o npnp_scored.csv

```
For a complete description of the available options use:
```
python -m pricePrediction.predict -h
```

##Retraining the model
###Preparing dataset

1) Split the Mcule catalogue file into train, test, and val .csv files and chunk them into smaller files.
   The raw Mcule catalogue file should contain, at least, the following columns:"Mcule ID,SMILES,price 1 (USD),amount 1 (mg),delivery time 1 (w.days),available amount 1 (mg)".
```
python -m  pricePrediction.selectFromRawData.selectFromRawDataNonVirtualAll --full_dataset_fname path/to/Mcule/dataset.csv --computed_datadir ./prepared_partions
```
```
e.g.

~/CoPriNet$ python -m  pricePrediction.selectFromRawData.selectFromRawDataNonVirtualAll --full_dataset_fname mcule_purchasable_full_prices_210616_O9Jw1D_only_READILY.csv.gz --computed_datadir ./prepared_partions
~/CoPriNet$ ls prepared_partions/
mcule_full_test.csv_split_0.csv    mcule_full_train.csv_split_10.csv  mcule_full_train.csv_split_22.csv  mcule_full_train.csv_split_34.csv
mcule_full_test.csv_split_1.csv    mcule_full_train.csv_split_11.csv  mcule_full_train.csv_split_23.csv  mcule_full_train.csv_split_35.csv
mcule_full_train.csv_split_00.csv  mcule_full_train.csv_split_12.csv  mcule_full_train.csv_split_24.csv  mcule_full_train.csv_split_36.csv
mcule_full_train.csv_split_01.csv  mcule_full_train.csv_split_13.csv  mcule_full_train.csv_split_25.csv  mcule_full_train.csv_split_37.csv
mcule_full_train.csv_split_02.csv  mcule_full_train.csv_split_14.csv  mcule_full_train.csv_split_26.csv  mcule_full_train.csv_split_38.csv
mcule_full_train.csv_split_03.csv  mcule_full_train.csv_split_15.csv  mcule_full_train.csv_split_27.csv  mcule_full_train.csv_split_39.csv
mcule_full_train.csv_split_04.csv  mcule_full_train.csv_split_16.csv  mcule_full_train.csv_split_28.csv  mcule_full_train.csv_split_40.csv
mcule_full_train.csv_split_05.csv  mcule_full_train.csv_split_17.csv  mcule_full_train.csv_split_29.csv  mcule_full_val.csv_split_0.csv
mcule_full_train.csv_split_06.csv  mcule_full_train.csv_split_18.csv  mcule_full_train.csv_split_30.csv  mcule_full_val.csv_split_1.csv
mcule_full_train.csv_split_07.csv  mcule_full_train.csv_split_19.csv  mcule_full_train.csv_split_31.csv  params.txt
mcule_full_train.csv_split_08.csv  mcule_full_train.csv_split_20.csv  mcule_full_train.csv_split_32.csv
mcule_full_train.csv_split_09.csv  mcule_full_train.csv_split_21.csv  mcule_full_train.csv_split_33.csv

```
Chunked files contain two columns, "SMILES,price" and will be named following the pattern:
```
RAW_DATA_FILE_SUFFIX = r"mcule_full_(train|test|val)\.csv_split_\w+\.csv$"
```
The file pattern can be edited changing the `pricePrediction/config.py` file. If you are using your own catalogue
you may want to split it manually into tran/test/validation and chunk the partitions into into smaller files. Each chunked
file should contain the header `SMILES,price` and the prices should be in $/g.

2) Create the dataset from the chunked .csv files that follow the pattern in the variable `pricePrediction.config.RAW_DATA_FILE_SUFFIX`
   The files should be csv files with two columns named SMILES,price

- Using the default paths included in `pricePrediction/config.py`:
    ```
    python -m  pricePrediction.preprocessData.prepareDataMol2Price
    ```
- Manually specifying the raw data chunks directory and the directory to store the prepared datasets. Set -n N to use N cpus.
    ```
    python -m  pricePrediction.preprocessData.prepareDataMol2Price -i /path/to/dir/with/csvFiles -o /path/to/save/prepared/data
    #e.g.  
    python -m  pricePrediction.preprocessData.prepareDataMol2Price -i ./prepared_partions -o ./dataset_encoded/ -n 32
    ```

3) Train the network
  ```
python -m pricePrediction.train.trainNet -m "one messege to store" --encodedDir /path/to/save/prepared/data
  
  e.g.
python -m pricePrediction.train.trainNet -m "trial01" --encodedDir ./dataset_encoded/ 
  ```
