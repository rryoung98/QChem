# QML
In this project we will work on making a highly extensible open source package focused on QCNN circuit templates and benchmarking on various frameworks and QC languages including Qiskit, Cirq/Tensorflow-Quantum, and Pennylane. 

The first phase of this project will involve making the QCNN templates in Pennylane. From there, we will use the Pennylane package to covert the CNN to Pytorch or Keras layers. 

Finally, we will create a benchmarking results function which will either benchmark the results for you _or_ in additon to F1 score, AUC, etc will printout some quantum metrics about the performance of the QCNN. (not sure yet)

We hope that this package will encompass other QML circuit templates which can be submitted then ran using Tensorflow / Pytorch as part of hybrid models. 

```
from QMLBenchmarker.layers import Mera, TTN 
import QMLBenchmarker
import tensorflow as tf
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.QCNN = Mera.torchLayer(100,200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.QCNN(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

### After Running

results = QMLBenchmarker.benchmark(model, type="pytorch")
results = QMLBenchmarker.benchmark(tensorflow, type="tensorflow")

# cool stats about the result visualized nicely

```

We'll make sure the benchmark results are rigorous taking inspiration from:
https://ogb.stanford.edu/docs/leader_overview/
