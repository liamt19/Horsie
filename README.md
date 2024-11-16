<h1 align="center">
Horsie - A C++ chess engine
</h1>

A (WIP) port of [Lizard](https://github.com/liamt19/Lizard/) from C# to C++. As of Nov. 15th, its search is (almost) functionally the same as [this commit](https://github.com/liamt19/Lizard/commit/3188b6eca26bc18d478d22d2d63931572d94e142).

---

[This test](http://somelizard.pythonanywhere.com/test/1985/) shows that of 5,108 pairs (both sides of a random opening), 5,105 were draws as expected. It also won 4 games, hence the "almost" functionally identical.
```
Elo   | 0.14 +- 0.16 (95%)
Conf  | N=5000 Threads=1 Hash=32MB
Games | N: 10216 W: 4189 L: 4185 D: 1842
Penta | [0, 0, 5105, 2, 1]
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
