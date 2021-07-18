# FinalProject

Made with Israel Cohen - another student, and supervised by prof. Ron Unger.

background:
Dr. Itay Koren from Bar-Ilan has conducted a biological reaserch and made data of 22,000 peptides - all 23-anino-acid-long, and a regressive stability score in range (0,6). He published his results here https://www.cell.com/cell/pdf/S0092-8674(18)30521-X.pdf
Based on this data, our project was constructing a machine that would learn to anticipate the stability score of peptides

getting started:
We decided to start with binary classification of 25% most stable and 25% least stable peptides, see how it goes, and later advance to regression.
Our initial intention for the binary classification was to start with MLP, get accuracy of around 60%, than move forward to RNN, becuse we assumed the order of AA in the sequenced plays a central role.

too good MLP results:
To take into account the order of AA within the sequence, we used one-hot-encoding: there are 20 different AA, so each AA in the peptide is ancoded to vector of 20, and overall peptides of 23 AA encoded to vector of 460.
To our great surprise, with the simple MLP we hit 95% accuracy!???
At first we suspected the results are to good to be true.
Indeed we noticed our data contains peptides that are very similar to one another, and some even identical (turned out the reasarch got two different results for the same peptides, it happens). Using cd-hit, we filtered the data from peptides that have 85% or more identical AA. The results remained the same, so we reduced it to 75%, and even 40%, with no significant change. We also tried using random half of the data (the data remained after 40% filtering was about half of the original) and got similar results. We concluded the problem we tried to solve - binary classification of 25% most stable and 25% least stable - is too easy, though it's weird that it can be solved with only MLP.
From now on we continued using the 40% filtered data, just to be sure.

simpler models:
At this point, we had a meeting with Dr. Itay who surprised us by agreeing that the order of AA within the sequence is not important. He showed us an article that stated that.
To make sure this is true, we tried ignoring the order by using bag-of-words encoding: each peptides was encoded to vector of 20, where vector[i] is how many times AA[i] exist in the peptide. The results where slightly better than before!???
After that we wanted to see how simpler models will do. We tried Linear Regression and Naive Bayes, with good results of ???

Regression:
Being satisfied with the classification, we moved on to regression. We tried Linear Regression??? and MLP regressor (from sklearn) ???
the results were similar using either one-hoe-encoding or bag-of-words encoding.

Proteins:
The peptides we used in fact are the tails of different proteins.
We wanted to find out what is the impact of the tail stability, to the stability of the whole protein.
We had a data file of the proteins, which contains the whole sequence, regressive stability score in range(???), and gene id, which we used to link each protein to its tail (in the data file of the peptides, they also have gene id).
To our regret, we succeeded to link only small portion of the proteins.
To determine the impact of the tail stability, we used MLP regressor to predict the stability of the protein, and used the following encoding: each protein was encoded to vector of 21, the first 20 are regular bag-of-words, and the last component is -1 for protein with unstable tail, 1 for stable, and 0 for the rest.
