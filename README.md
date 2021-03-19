# muzero

Trying to use muzero

Learning how to use muzero to play cartpole-v1 or lunarlander-v2 gym games

To run follow these steps:

## Clone and Create Virtual Env

```
git clone https://github.com/ipsec/muzero.git
cd muzero
python3.8 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run script with

```
python muzero.py
```

## See progress in tensorboard

```
tensorboard --logdir data
```
