{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6931e289",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "\n",
    "class NeuralNet(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, num_classes):\n",
    "        super(NeuralNet, self).__init__()\n",
    "        self.l1 = nn.Linear(input_size, hidden_size) \n",
    "        self.l2 = nn.Linear(hidden_size, hidden_size) \n",
    "        self.l3 = nn.Linear(hidden_size, num_classes)\n",
    "        self.relu = nn.ReLU()\n",
    "    \n",
    "    def forward(self, x):\n",
    "        out = self.l1(x)\n",
    "        out = self.relu(out)\n",
    "        out = self.l2(out)\n",
    "        out = self.relu(out)\n",
    "        out = self.l3(out)\n",
    "        # no activation and no softmax at the end\n",
    "        return out"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
