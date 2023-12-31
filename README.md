# Portfolio
Die sechs Projekte sollen einen diversen Einblick in die verschiedenen Themengebiete meiner bisherigen Ausbildung geben.

### 01 Custom MLP Implementierung
In diesem Projekt wurde ein eigener Multi Layer Perceptron ( MLP ) von Grund auf selbst implementiert. Für die Implementierung wurden nur SciPy und NumPy verwendet. 

Wird main.py (requirements.txt beachten) ausgeführt, wird der tuned MLP mit dem Mice Protein Expression (https://archive.ics.uci.edu/dataset/342/mice+protein+expression) Dataset trainiert und evaluiert. Das f1-Score, Accuracy und Runtime werden in der Konsole ausgegeben. 

Es kann auch Grid Search Hyperparameter Tuning über ein Custom Evaluation Model in drei verschiedenen Datasets ausgeführt werden. 

In plots.png ist der Vergleich mit der Implementierung von Sci-Kit Learn zu sehen.
### 02 Breakout Reinforcement Learning
In diesem Projekt wurde das Spiel Breakout mithilfe von Monte Carlo-basiertem Reinforcement Learning automatisiert. Der Agent wird ohne vorheriges Wissen trainiert und findet die schnellste Lösung. Das Spiel des Agenten wird mittels PyGame visualisiert.

Weitere Informationen im detailierten ReadMe.

### 03 Monte Carlo Ising Simulation
In diesem Projekt wird das Ising Modell mithilfe von Monte Carlo-Simulation mit dem Metropolis-Verfahren auf dem Quadratgitter mit Nächst-Nachbar-Wechselwirkung simuliert.

Auszuführen mit run.py. Im img Ordner ist die gefundene Binder-Kumulante und die Magnetsierung in Abhängigkeit von der Temperatur geplottet.

### 04 Simulated-Annealing von NMR-Brain Scans

In diesem Projekt wird ein NMR-Gehirnbild mithilfe von Simulated Annealing mit a priori Nachbarschaftskorrelation zu den 4 nächsten Nachbarn segmentiert.

Auszuführen indem run.py gestartet wird. In den beiden .png-Dateien sind die segmentierten Ergebnisse zu sehen.

### 05 Low-rank approximations in the Ising model

Jupyter Notebook in dem thin SVD und truncated SVD sowie Principal Component Analysis verwendet wird um Eigenschaften des Ising Modells zu untersuchen.

### 06 Classifying Supersymmetry using Logistic Regression

Jupyter Notebook in dem Daten (5.000.000 Events, 18 Variablen) aus dem Large Hadron Collider mithilfe von Logistic Regression klassifiziert werden.


