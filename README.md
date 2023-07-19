# NA_NMT

## Abstract 

Non-Autoregressive models are widely used in the field of Neural Machine Translation. 
We employ three different NA transformer CMLM models with the use of two variations of Reward Based loss functions and we compare our results with
an implemented NA baseline model with Cross Entropy loss.
In addition we set side by side our BLEU and ChrF scores with an AR baseline model and other similar NA MRT models from the literature.

## Main Contribution

We train a Non-Autoregressive translation model with CMLM and the use of MRT as a loss function. We pick BLUE and ChrF as sentence level evaluation metrics that will provide a score based on how good is the translation quality of our predicted sentence. We compare our results with other NA model approaches and settings described more extensively in the Related Work section. As the main baseline we use an NA model trained with cross entropy as a cost function. 
Our fine tuned CMLM model performs worse than the baseline in terms of both aforementioned metrics.

Our work focuses on whether MRT provides a sense of an autoregressive approach to our NA model and boosts its performance. Which are the most appropriate metrics to be used and under what scenarios. Lastly, an interesting research question would be whether a simultaneous use of various rewards in the loss function of a CMLM MRT model  may boost its performance. 

## Code Base
Insructions regarding code reproducibility are provided in nltk_blue_mini_projectA.ipynb python notebook.
