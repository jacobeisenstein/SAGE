SAGE in Python
==========

You can run SAGE in Python or from the command line to identify keywords that distinguish parts of a corpus of text. For example, SAGE can be used to distinguish:

- tweets from [different places](http://www.cc.gatech.edu/~jeisenst/papers/dialectology-chapter.pdf) or [across time](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113114)
- political speech by party or by vote
- interests and speaking styles across subreddits

These instructions show you how to use SAGE from the command line.

[This notebook](using-sage.ipynb) shows you how to use SAGE in a Jupyter notebook.

If you use SAGE, please cite [the paper](http://www.icml-2011.org/papers/534_icmlpaper.pdf):

```Eisenstein, Jacob, Amr Ahmed, and Eric P. Xing. "Sparse Additive Generative Models of Text." Proceedings of the 28th International Conference on Machine Learning (ICML-11). 2011.```

# Congressional votes

The Cornell [convote](http://www.cs.cornell.edu/home/llee/data/convote.html) data includes floor speeches from the U.S. Congress. I have extracted a subset of speeches about an appropriations bill. Speeches in favor of the bill are in ```convote-132-yes.txt``` and speeches against are in ```convote-132-no.txt```.

To run SAGE to find keywords from each set:

```python runSage.py "convote-132-*" --max_vocab_size 1000 --base_rate_smoothing=1. --num_keywords=5```

Here are the arguments:

- ```"convote-132-*``` This specifies a file glob to analyze.
- ```--max_vocab_size 1000``` This indicates that SAGE will consider only the 1000 most frequent words.
- ```--base_rate_smoothing=1.``` This smooths the baseline language model. Lower values will cause SAGE to emphasize more rare words; higher values will cause it emphasize more frequent words.
- ```--num_keywords=5``` This is the number of keywords to output.

The output is in the form of a TSV file, which I converted to Markdown using [this script](https://donatstudios.com/CsvToMarkdownTable).

|                     |            |                     |            |                        |            |                        | 
|---------------------|------------|---------------------|------------|------------------------|------------|------------------------| 
| source              | word       | sage                | base_count | base_rate              | file_count | file_rate              | 
| convote-132-yes.txt | initiative | 0.64485977013076057 | 32         | 0.00023553138087632395 | 30         | 0.00048760666395774075 | 
| convote-132-yes.txt | corps      | 0.59385906984348791 | 119        | 0.00087588232263382962 | 100        | 0.0016253555465258025  | 
| convote-132-yes.txt | fertilizer | 0.58452194616875175 | 24         | 0.00017664853565724296 | 22         | 0.00035757822023567654 | 
| convote-132-yes.txt | request    | 0.54996109160370776 | 63         | 0.00046370240610026277 | 52         | 0.00084518488419341729 | 
| convote-132-yes.txt | weapons    | 0.54407401636419572 | 30         | 0.00022081066957155369 | 26         | 0.00042259244209670864 | 
| convote-132-no.txt  | profits    | 0.53036748717874982 | 39         | 0.0002870538704430198  | 39         | 0.00052463074067098927 | 
| convote-132-no.txt  | drinking   | 0.52818311966607179 | 65         | 0.00047842311740503303 | 63         | 0.00084748042723775186 | 
| convote-132-no.txt  | taxpayers  | 0.51075530349671516 | 54         | 0.00039745920522879665 | 52         | 0.00069950765422798573 | 
| convote-132-no.txt  | taxpayer   | 0.48954718467824154 | 36         | 0.00026497280348586441 | 35         | 0.00047082245957652882 | 
| convote-132-no.txt  | bush       | 0.48657112756142901 | 71         | 0.0005225852513193437  | 66         | 0.0008878366380585972  | 

# Second example

Todo: a second example, using the ```basefile``` command line argument. Need to find a good dataset for this.
