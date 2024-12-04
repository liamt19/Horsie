<h1 align="center">
Horsie - A C++ chess engine
</h1>

A (WIP) port of [Lizard](https://github.com/liamt19/Lizard/) from C# to C++. As of December 4th, its search is functionally identical to [this commit](https://github.com/liamt19/Lizard/commit/5cfcc4f7fb0540bda504460a0225a85a44bc2c74).

---

[This test](http://somelizard.pythonanywhere.com/test/2016/) shows that of 5,155 pairs (both sides of a random opening), all were draws as expected.
```
Elo   | -0.00 +- 0.00 (95%)
Conf  | N=5000 Threads=1 Hash=16MB
Games | N: 10310 W: 4342 L: 4342 D: 1626
Penta | [0, 0, 5155, 0, 0]
```


---

I did this to challenge myself to translate a decent-sized project, and also to test the difference in performance between the two languages.

[This test](http://somelizard.pythonanywhere.com/test/1986/) was done at a short time control, in which speed has the biggest impact:
```
Elo   | 26.27 +- 6.95 (95%)
SPRT  | 8.0+0.08s Threads=1 Hash=32MB
LLR   | 2.99 (-2.94, 2.94) [0.00, 3.00]
Games | N: 2412 W: 656 L: 474 D: 1282
Penta | [5, 193, 636, 359, 13]
```
