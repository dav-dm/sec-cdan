# Hyperparameter Configurations for Training Approaches

This folder contains `.json` files with the hyperparameters used to train the various approaches presented in the [paper]().
It also includes the file `dataset_sample.parquet`, which serves as an example of the required dataset format for use with this framework.

---

## Example Commands to Run Different Approaches

### ML Approaches

* **Label Spreading**  
```bash
python
````

* **Label Propagation**

```bash
python 
```

### DL Approaches

* **NoAdapt**

```bash
python 
```

* **MCC**

```bash
python 
```

* **ADDA**

```bash
python 
```

* **Sec-CDAN**

```bash
python main.py --src-dataset cic2018 --trg-dataset insdn --approach sec_cdan --seed 0 --log-dir ADD_A_PATH --max-epochs 200 --network transformer --adapt-epochs 200 --n-tasks 2 --adapt-batch-size 128 --iter-per-epoch 150 --discr-hidden-size 50 --cdan-alpha 2
```

---

Replace `ADD_A_PATH` with the actual paths to your log directories or saved models.
