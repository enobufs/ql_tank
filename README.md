# ql_tank
Simple reinforcement learning using Q-Learning.

This program impolements Q-Learning algorithm explained in the following awesome video lecture by [Dr. Jacob Schrum](https://people.southwestern.edu/~schrum2/):

* Reinforcement Learning 1 - Expected Values ([youtube](https://www.youtube.com/watch?v=3T5eCou2erg))
* Reinforcement Learning 2 - Grid World ([youtube](https://www.youtube.com/watch?v=bHeeaXgqVig))
* Reinforcement Learning 3 - Q Learning ([youtube](https://www.youtube.com/watch?v=1XRahNzA5bE))
* Reinforcement Learning 4 - Q Learning Parameters ([youtube](https://www.youtube.com/watch?v=XrxgdpduWOU))

This code implements the following Q-Learning equation:

![math](https://latex.codecogs.com/gif.latex?Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%20%2B%3D%20%5Calpha%20%5Cleft%20%5C%7B%20r_%7Bt%2B1%7D%20%2B%20%5Cgamma%20%5Ccdot%20%5Cmax_%7Ba%7DQ%28s_%7Bt%2B1%7D%2C%20a%29-Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%20%5Cright%20%5C%7D)


## Requirments
* Python3
* Dependencies: numpy, matplotlib, readchar

## How to Run

### Manual actions
The manual mode would help you understand how Q(s, a) values are updated as you move.

```shell
$ python3 ql_tank.py

Auto-training:  False
>>> Use arrow keys to perform actions. Hit other keys to exit. <<<
EPISODE: 1, MOVE: 0, ERROR RATE: 0.00
+-------------+-------------+-------------+-------------+
|    +0.00    |    +0.00    |    +0.00    |+-----------+|
|+0.00   +0.00|+0.00   +0.00|+0.00   +0.00||  Success  ||
|    +0.00    |    +0.00    |    +0.00    |+-----------+|
+-------------+-------------+-------------+-------------+
|    +0.00    |#############|    +0.00    |+-----------+|
|+0.00   +0.00|#############|+0.00   +0.00||    Fail   ||
|    +0.00    |#############|    +0.00    |+-----------+|
+-------------+-------------+-------------+-------------+
|    +0.00    |    +0.00    |    +0.00    |    +0.00    |
|+0.00 * +0.00|+0.00   +0.00|+0.00   +0.00|+0.00   +0.00|
|    +0.00    |    +0.00    |    +0.00    |    +0.00    |
+-------------+-------------+-------------+-------------+
```

The '*' indicates where the current position is. Use arrow keys to move. The cell filled with `#` is
a wall, and you cannot enter.


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
