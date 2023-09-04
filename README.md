# Portfolio
Die vier Projekte sollen einen diversen Einblick in die verschiedenen Themengebiete meiner bisherigen Ausbildung geben.

### 01 Custom MLP Implementierung
In diesem Projekt wurde ein eigener Multi Layer Perceptron ( MLP ) von Grund auf selbst implementiert. Für die Implementierung wurden nur SciPy und NumPy verwendet. 

Wird main.py (requirements.txt beachten) ausgeführt, wird der tuned MLP mit dem # Mice Protein Expression (https://archive.ics.uci.edu/dataset/342/mice+protein+expression) Dataset trainiert und evaluiert. Das Scoring wird ausgegeben. 

Es kann auch Grid Search Hyperparameter Tuning über ein Custom Evaluation Model in drei verschiedenen Datasets ausgeführt werden. 

In plots.png ist der Vergleich mit der Implementierung von Sci-Kit Learn zu sehen.
### 02 Breakout Reinforcement Learning
In diesem Projekt wurde das Spiel Breakout mithilfe von Monte Carlo basiertem Reinforcement Learning automatisiert. Der Agent wird ohne vorheriges Wissen trainiert und findet die schnellste Lösung. Das Spiel des Agenten wird mittels PyGame visualisiert.

Weitere Informationen im detailierten ReadMe.

### 03 Monte Carlo Ising Simulation
In diesem Projekt wird das Ising Modell mithilfe von Monte-Carlo-Simulation mit dem Metropolis-Verfahren auf dem Quadratgitter mit Nächst-Nachbar-Wechselwirkung simuliert.

Auszuführen mit run.py. Im img Ordner ist die gefundene Binder Kumulante und die Magnetsierung in Abhängigkeit von der Temperatur geplottet.

### 04 Simulated-Annealing von NMR-Brain Scans

In diesem Projekt wird ein NMR Gehirn Bild mithilfe von Simulated Annealing mit *a priori* Nachbarschaftskorrelation zu den 4 nächsten Nachbarn segmentiert.

Auszuführen indem run.py gestartet wird. In den beiden .png Dateien sind die Segmentierten Ergebnisse zu sehen.
