# N-Gram Language Model Implementation

This was an assignment for CS 4120, Natural Language Processing, at Northeastern University.

## Description
This model can generate the probability of a given whitespace delimited sequence in which the first token is a sentence start token (`"<s>"`), the last token is a sentence end token (`"</s>"`), and there is no punctuation.

In addition to this, there is a `perplexity` method in the `LanguageModel` class which can compute this model's perplexity for a given string sequence, with the same constraints as above.

## General Usage
```
python lm.py trainingfile.txt testfile_1.txt testfile_2.txt ... testfile_n.txt
```

**In order to change the model(s) instantiated, the `languageModelConfiguration` must be modified with another tuple that contains the parameters to the `LanguageModel` class. This includes the value of `n` and the toggle for the use of Laplace smoothing.**

## Example Usage & Output

```
> cd N-Gram-Language-Model
> python lm.py berp-training.txt test1.txt
model: n = 1 , laplacedSmoothed = True
sentences (first 10 lines):
<s> let's start over </s>
<s> my mother is coming to visit and i'd like to take her to dinner </s>
<s> new query </s>
<s> now i'm interested in some middle eastern food </s>
<s> oh i have to breakfast  </s>
<s> oh i increase the walking distance i can go fifteen minutes from icsi </s>
<s> oh i would like to have french food  </s>
<s> okay back to the normal stuff what about mexican </s>
<s> okay uh how about american or french or uh european food  </s>
<s> prefer german food </s>
test set: test1.txt
number of test sentences: 107
average probability: 2.16673717680423e-14
standard deviation: 1.2306022254777805e-13
perplexity of test1.txt : 0.42074863777531124

model: n = 2 , laplacedSmoothed = True
sentences (first 10 lines):
<s> let's start over </s>
<s> my mother is coming to visit and i'd like to take her to dinner </s>
<s> new query </s>
<s> now i'm interested in some middle eastern food </s>
<s> oh i have to breakfast  </s>
<s> oh i increase the walking distance i can go fifteen minutes from icsi </s>
<s> oh i would like to have french food  </s>
<s> okay back to the normal stuff what about mexican </s>
<s> okay uh how about american or french or uh european food  </s>
<s> prefer german food </s>
test set: test1.txt
number of test sentences: 107
average probability: 8.139126116514245e-11
standard deviation: 8.337042690320485e-10
perplexity of test1.txt : 0.39203677868840164

```