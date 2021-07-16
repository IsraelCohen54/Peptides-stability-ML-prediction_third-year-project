# FinalProject

Made with Israel Cohen - another student, and supervised by prof. Ron Unger.

background:
Dr Itay Koren from Bar-Ilan has conducted a biological reaserch and made data of 22,000 peptides - all 23-anino-acid-long, and a stability score in range (0,6). He published his results here https://www.cell.com/cell/pdf/S0092-8674(18)30521-X.pdf
Based on this data, our project was constructing a machine that would learn to anticipate the stability score of peptides

getting started:
We decided to start with binary classification of 25% most stable and 25% least stable peptides, see how it goes, and later advance to regression.
Our initial intention for the binary classification was to start with MLP, get accuracy of around 60%, than move forward to RNN, becuse we assumed the order of AA in the sequenced plays a central role.
To our great surprise, with the simple MLP we hit 95% accuracy!

to to 
