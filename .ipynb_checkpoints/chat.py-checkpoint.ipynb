{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b400b79",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import json\n",
    "\n",
    "import torch\n",
    "\n",
    "from model import NeuralNet\n",
    "from nltk_utils import bag_of_words, tokenize\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "with open('intents.json', 'r') as json_data:\n",
    "    intents = json.load(json_data)\n",
    "\n",
    "FILE = \"trained.pth\"\n",
    "data = torch.load(FILE)\n",
    "\n",
    "input_size = data[\"input_size\"]\n",
    "hidden_size = data[\"hidden_size\"]\n",
    "output_size = data[\"output_size\"]\n",
    "all_words = data['all_words']\n",
    "tags = data['tags']\n",
    "model_state = data[\"model_state\"]\n",
    "\n",
    "model = NeuralNet(input_size, hidden_size, output_size).to(device)\n",
    "model.load_state_dict(model_state)\n",
    "model.eval()\n",
    "\n",
    "bot_name = \"Lennie\"\n",
    "print(\"Let's have a chat! (type 'exit' to stop chatting!)\")\n",
    "\n",
    "while True:\n",
    "    sentence = input(\"You: \")\n",
    "    if sentence == \"exit\":\n",
    "        print(\"Thanks for joining me!\")\n",
    "        break\n",
    "\n",
    "    sentence = tokenize(sentence)\n",
    "    X = bag_of_words(sentence, all_words)\n",
    "    X = X.reshape(1, X.shape[0])\n",
    "    X = torch.from_numpy(X).to(device)\n",
    "\n",
    "    output = model(X)\n",
    "    _, predicted = torch.max(output, dim=1)\n",
    "\n",
    "    tag = tags[predicted.item()]\n",
    "\n",
    "    probs = torch.softmax(output, dim=1)\n",
    "    prob = probs[0][predicted.item()]\n",
    "    print(prob)\n",
    "    if prob.item() > 0.75:\n",
    "        for intent in intents['intents']:\n",
    "            if tag == intent['tag']:\n",
    "                print(f\"{bot_name}: {random.choice(intent['responses'])}\")\n",
    "    else:\n",
    "        print(f\"{bot_name}: I do not understand. Try to be more specific\")"
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
