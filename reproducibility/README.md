# Hyperparameter Configurations for Training Approaches

This folder contains the `.json` files with the hyperparameters used to train the different approaches presented in the [paper]().

It also includes the file `dataset_sample.parquet`, which provides an example of the dataset format required by the framework.

---

## Example Commands to Run the Approaches

Below are example commands to run each approach.

### ML Approaches

* **Label Spreading**  
```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach label_spreading --seed 0 --log-dir ADD_A_PATH --is-flat --n-tasks 1
````

* **Label Propagation**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach label_propagation --seed 0 --log-dir ADD_A_PATH --is-flat --n-tasks 1 
```

### DL Approaches

* **NoAdapt**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach baseline --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --adapt-epochs 200 --network transformer --n-tasks 2 --skip-t2 
```

* **MCC**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach mcc --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --network transformer --adapt-epochs 200 --n-tasks 2 --adapt-batch-size 128 --iter-per-epoch 150 
```

* **ICON**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach icon --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --network transformer --adapt-epochs 200 --n-tasks 2 --adapt-batch-size 64 --iter-per-epoch 50  
```

* **ADDA**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach adda --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --network transformer --adapt-epochs 200 --n-tasks 2 --adapt-batch-size 128 --iter-per-epoch 50  --discr-hidden-size 250 
```

* **Sec-CDAN**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach sec_cdan --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --network transformer --adapt-epochs 200 --n-thr 1 --n-tasks 2 --adapt-batch-size 128 --iter-per-epoch 150 --discr-hidden-size 50 --cdan-alpha 2
```

---

Replace `ADD_A_PATH` with the actual paths to your log directories.
