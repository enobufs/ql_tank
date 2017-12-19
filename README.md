# ql_tank
Simple reinforcement learning using Q-Learning.

This program impolements Q-Learning algorithm explained in the following awesome YouTube tutorials:
* Reinforcement Learning 1 - Expected Values ([youtube](https://www.youtube.com/watch?v=3T5eCou2erg))
* Reinforcement Learning 2 - Grid World ([youtube](https://www.youtube.com/watch?v=bHeeaXgqVig))
* Reinforcement Learning 3 - Q Learning ([youtube](https://www.youtube.com/watch?v=1XRahNzA5bE))
* Reinforcement Learning 4 - Q Learning Parameters ([youtube](https://www.youtube.com/watch?v=XrxgdpduWOU))

## Pre-requisite
* Python3
* pip3 install matplotlib

## How to Run

### Manual actions
The manual mode would help you understand how Q(s, a) values are updated as you move.

```shell
$ python3 ql_tank.py

Auto-training:  False
>>> Use arrow keys to perform actions. Hit enter to exit. <<<
EPISODE: 1, MOVE: 0, ERROR RATE: 0.00
+-------------+-------------+-------------+-------------+
|             |             |             |             |
|      ?      |      ?      |      ?      |      ?      |
|             |             |             |             |
+-------------+-------------+-------------+-------------+
|             |#############|             |             |
|      ?      |#############|      ?      |      ?      |
|             |#############|             |             |
+-------------+-------------+-------------+-------------+
|    +0.00    |             |             |             |
|+0.00 * +0.00|      ?      |      ?      |      ?      |
|    +0.00    |             |             |             |
+-------------+-------------+-------------+-------------+
```

The '*' indicates where the current position is. Use arrow keys to move. The cell filled with `#` is
a wall, and you cannot enter. The cells with `?` implies that you (the agent) does not know what is
there yet.


### Automatic actions

```shell
$ python3 ql_tank.py --auto
```

The iteration by default is set to 400. To modify:

```shell
$ python3 ql_tank.py --auto --episodes 1000
```

When all episodes complete, it will plot the error ratio, using matplotlib.


## Note
* Tested only with MBP, python 3.6.3.
