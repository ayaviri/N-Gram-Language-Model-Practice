import sys
import math
import numpy
import random
import statistics

# name: joshua ayaviri

class LanguageModel:
  UNK = "<UNK>"
  SENT_BEGIN = "<s>"
  SENT_END = "</s>"

  def __init__(self, n: int, usesLaplaceSmoothing: bool):
    """Initializes an untrained LanguageModel
    Parameters:
      n (int): the n-gram order of the language model to create
      usesLaplaceSmoothing (bool): whether or not to use Laplace smoothing
    """
    self.n = n
    # the number of sentence start/end tokens to add on each end of the sequence
    self.numberOfPads = self.n - 2
    self.usesLaplaceSmoothing = usesLaplaceSmoothing
    # a dictionary that maps from a unique type to its count in the training corpus
    self.vocabulary = {}
    # a dictionary that maps from a unique n-gram to its count in the training corpus
    self.nGrams = {}
    # a dictionary tha tmaps from a unique (n-1)-gram to its count in the training corpus (used in the denominator of each n-grams probability)
    self.nMinusOneGrams = {}
    self.tokens = 0
    # a list of n-grams for sentence generation using Shannon's method
    self.generationList = []

  def __padSequence(self, sequence):
    """Converts a sentece to a list of tokens, padded by the N value of this language model
    Parameters:
      sequence (str): The sequence prior to tokenization

    Returns:
    The tokenized sequence with padding
    """
    tokens = sequence.split()
    for index in range(self.numberOfPads):
          tokens = [self.SENT_BEGIN, *tokens]
          tokens += [self.SENT_END]
    return tokens

  def train(self, trainingFilePath):
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      trainingFilePath (str): the location of the training data to read

    Returns:
    None
    """
    self.vocabulary = {}
    self.nGrams = {}
    self.nMinusOneGrams = {}
    self.tokens = 0
    self.generationList

    with open(trainingFilePath) as trainingFile:
      sentences = trainingFile.readlines()
      # a temporary dictionary so that keys can be removed from it during iteration
      beforePreprocessingVocabulary = {}

      # iterate through the first time to find types with count 1
      for sentence in sentences:
        currentTokens = sentence.split()

        for token in currentTokens:
          self.tokens += 1
          if token in self.vocabulary:
            self.vocabulary[token] += 1
            beforePreprocessingVocabulary[token] += 1
          else:
            self.vocabulary[token] = 1
            beforePreprocessingVocabulary[token] = 1
      
      # here, we replace single count types with unk
      unknownCount = 0
      for type in beforePreprocessingVocabulary:
        # remove from self.vocabulary and increase unknownCount by 1
        if beforePreprocessingVocabulary[type] == 1:
          self.vocabulary.pop(type)
          unknownCount += 1

      if unknownCount > 0:
        self.vocabulary[self.UNK] = unknownCount

      # now we can populate nGram dictionary
      for sentence in sentences:
        currentTokens = self.__padSequence(sentence)
        
        # now we're ready to populate n-gram dictionary
        currentNumberOfNGrams = len(currentTokens) - (self.n - 1)
        for index in range(currentNumberOfNGrams):
          currentNGram = currentTokens[index:index + self.n]
          
          for tokenIndex in range(len(currentNGram)):
            currentToken = currentNGram[tokenIndex]
            if not currentToken in self.vocabulary:
              currentNGram[tokenIndex] = self.UNK

          # conversion to tuple so that it is hashable
          currentNGram = tuple(currentNGram)

          # addition/edit to n-gram dictionary
          if currentNGram in self.nGrams:
            self.nGrams[currentNGram] += 1
          else:
            self.nGrams[currentNGram] = 1

        # finally, let's populate the (n-1)-gram dictionary
        currentNumberOfNMinusOneGrams = len(currentTokens) - (self.n - 2)
        for index in range(currentNumberOfNMinusOneGrams):
          currentNMinusOneGram = tuple(currentTokens[index:index + (self.n - 1)])

          # addition/edit to (n-1)-gram dictionary
          if currentNMinusOneGram in self.nMinusOneGrams:
            self.nMinusOneGrams[currentNMinusOneGram] += 1
          else:
            self.nMinusOneGrams[currentNMinusOneGram] = 1

  def score(self, sentence):
    """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """
    # conversion to a list of tokens, padded given n
    tokens = self.__padSequence(sentence)

    # let us replace unknown words with the UNK tag
    for index in range(len(tokens)):
      currentToken = tokens[index]
      if not currentToken in self.vocabulary:
        tokens[index] = self.UNK

    totalLogProbability = 0
    currentNumberOfNGrams = len(tokens) - (self.n - 1)
    for index in range(currentNumberOfNGrams):
      currentNGram = tuple(tokens[index:index + self.n])
      currentNMinusOneGram = tuple(tokens[index:index + self.n - 1])

      if self.n == 1:
        numerator = self.nGrams[currentNGram] if currentNGram in self.nGrams else 0
        numerator += 1 if self.usesLaplaceSmoothing else 0
        denominator = self.tokens + len(self.vocabulary) if self.usesLaplaceSmoothing else self.tokens
      else:
        # it is here where we'll smooth using laplace smoothing
        if self.usesLaplaceSmoothing:
          numerator = self.nGrams[currentNGram] + 1 if currentNGram in self.nGrams else 1  
          denominator = self.nMinusOneGrams[currentNMinusOneGram] + len(self.vocabulary) if currentNMinusOneGram in self.nMinusOneGrams else len(self.vocabulary)
        else:
          numerator = self.nGrams[currentNGram] if currentNGram in self.nGrams else 0
          denominator = self.nMinusOneGrams[currentNMinusOneGram]

      totalLogProbability += numpy.log(numerator / denominator)

    return math.e ** totalLogProbability
    
  def generate_sentence(self):
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
    endIndex = self.n - 1 if self.n != 1 else 1
    # this list will control the possible n-grams for a given iteration of this method
    sampleSpace = []
    for nGram in self.generationList:
      if list(nGram[0:endIndex]) == [self.SENT_BEGIN] * endIndex:
        sampleSpace += [nGram]

    # because the given iterable must be one-dimensional, we'll choose a random index instead
    firstIndex = random.randrange(len(sampleSpace))
    firstNGram = sampleSpace[firstIndex]
    sentence = list(firstNGram)

    # the last n-1 words of the n gram
    desiredPrefix = firstNGram[1:]
    # last word of n gram
    currentSuffix = firstNGram[-1]

    while currentSuffix != self.SENT_END:
      # 1) reconstruct sample space
      sampleSpace = []
      if self.n != 1:
        for nGram in self.generationList:
          if nGram[0:endIndex] == desiredPrefix:
            sampleSpace += [nGram]
      else:
        for nGram in self.generationList:
          if nGram[0] != self.SENT_BEGIN:
            sampleSpace += [nGram]

      # 2) choose from it
      index = random.randrange(len(sampleSpace))
      nGram = sampleSpace[index]

      # 3) add the last element of n-gram to sentence
      # 4) reset the last word added
      desiredPrefix = nGram[1:]
      currentSuffix = nGram[-1]

      sentence += [currentSuffix]

    # padding the end of the sentence with the necessary end of sentence tags
    for index in range(self.numberOfPads):
      sentence += [self.SENT_END]
    return " ".join(sentence)

  def generate(self, n):
    """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
    sentences = []

    # this essentially recreates the corpus
    if len(self.generationList) == 0:
      for nGram in self.nGrams:
        count = self.nGrams[nGram]
        for index in range(count):
          self.generationList += [nGram]

    sentences = []
    for index in range(n):
      currentSentence = self.generate_sentence()
      sentences += [currentSentence]
    return sentences

  def perplexity(self, test_sequence):
    # force perplexity calculation to use laplace smoothing
    oldValue = self.usesLaplaceSmoothing
    self.usesLaplaceSmoothing = True
    sequenceProbability = self.score(test_sequence)
    self.usesLaplaceSmoothing = oldValue
    perplexity = (1 / sequenceProbability) ** (-1 / len(test_sequence))
    return perplexity

def main():
  trainingPath = sys.argv[1]
  testingPaths = sys.argv[2:2+2]

  trainingSentences = []
  with open(trainingPath) as file:
    lines = file.readlines()
    for index in range(len(lines)):
      trainingSentences += [lines[index][:-1]]

  languageModelConfiguration = [(1, True), (2, True)]

  for configuration in languageModelConfiguration:
    currentModel = LanguageModel(configuration[0], configuration[1])
    testSets = []

    for testingPath in testingPaths:
      currentTestSet = []
      with open(testingPath) as file:
        lines = file.readlines()

        # getting rid of the newline character at the end of each line
        for index in range(len(lines)):
          strippedLine = lines[index].strip()
          if strippedLine != "":
            currentTestSet += [strippedLine]
      testSets += [currentTestSet]

    print("model: n =", configuration[0], ", laplacedSmoothed =", configuration[1])

    # given that the training corpus is huge, i'll only print the first 10 lines
    linesToPrint = 10
    print("sentences (first", linesToPrint, "lines):")
    for index in range(len(trainingSentences)):
      if index >= linesToPrint:
        break
      print(trainingSentences[index])
    
    # now we print results for each test set
    currentModel.train(trainingPath)
    for index in range(len(testSets)):
      print("test set:", testingPaths[index])
      print("number of test sentences:", len(testSets[index]))
      probabilitySum = 0
      testProbabilities = []
      for sequence in testSets[index]:
        currentProbability = currentModel.score(sequence)
        probabilitySum += currentProbability
        testProbabilities += [currentProbability]
      print("average probability:", probabilitySum / len(testProbabilities))
      print("standard deviation:", statistics.stdev(testProbabilities))

      # the number of sentences to glue together for perplexity calculation
      numberOfSentences = 10
      perplexitySequence = " ".join(testSets[index][0:numberOfSentences])
      print("perplexity of", testingPaths[index], ":", currentModel.perplexity(perplexitySequence))
    print()

if __name__ == '__main__':
    
  # first argument must be training file, each subsequent one must be a test file
  if len(sys.argv) < 3:
    print("Usage:", "python lm.py training_file.txt testingfile_1.txt testingfile_2.txt ... testingfile_n.txt")
    sys.exit(1)

  main()

