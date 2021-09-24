{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e13f0e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import nltk\n",
    "# nltk.download('punkt')\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "stemmer = PorterStemmer()\n",
    "\n",
    "def tokenize(sentence):\n",
    "    \"\"\"\n",
    "    split sentence into array of words/tokens\n",
    "    a token can be a word or punctuation character, or number\n",
    "    \"\"\"\n",
    "    return nltk.word_tokenize(sentence)\n",
    "\n",
    "\n",
    "def stem(word):\n",
    "    \"\"\"\n",
    "    stemming = find the root form of the word\n",
    "    examples:\n",
    "    words = [\"organize\", \"organizes\", \"organizing\"]\n",
    "    words = [stem(w) for w in words]\n",
    "    -> [\"organ\", \"organ\", \"organ\"]\n",
    "    \"\"\"\n",
    "    return stemmer.stem(word.lower())\n",
    "\n",
    "\n",
    "def bag_of_words(tokenized_sentence, words):\n",
    "    \"\"\"\n",
    "    return bag of words array:\n",
    "    1 for each known word that exists in the sentence, 0 otherwise\n",
    "    example:\n",
    "    sentence = [\"hello\", \"how\", \"are\", \"you\"]\n",
    "    words = [\"hi\", \"hello\", \"I\", \"you\", \"bye\", \"thank\", \"cool\"]\n",
    "    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]\n",
    "    \"\"\"\n",
    "    # stem each word\n",
    "    sentence_words = [stem(word) for word in tokenized_sentence]\n",
    "    # initialize bag with 0 for each word\n",
    "    bag = np.zeros(len(words), dtype=np.float32)\n",
    "    for idx, w in enumerate(words):\n",
    "        if w in sentence_words: \n",
    "            bag[idx] = 1\n",
    "\n",
    "    return bag"
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
