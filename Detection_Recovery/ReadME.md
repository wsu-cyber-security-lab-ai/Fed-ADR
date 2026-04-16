
## 1.  Requirements

- PyTorch  
- Torchvision  
- NumPy  
- scikit-learn  
- matplotlib  
- tabulate  

---

## 2.  Create Environment (Recommended)

```bash
conda create -n fedrecover python=3.9 -y
conda activate fedrecover
```

Or:

```bash
python -m venv fedrecover_env
source fedrecover_env/bin/activate
```

---

## 3.  Install Dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib tabulate
```

---

## 4.  Project Structure

```
project/
│── main.py
│── data/
```

---

## 5.  Important Path Fix

Update:

```python
local_folder = "Historacal data PATH"
```

To:

```python
local_folder = "./data_logs"
```

---

## 6.  Run the Code

```bash
python main.py
```

---

## 7.  Optional Arguments

```bash
python main.py --lr 0.001 --nworkers 20 --nepochs 30 --byz_type gaussian
```


