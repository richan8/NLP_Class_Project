# Reddit User classifier

Project for Fall 2020 Practical Natural Language Processing 9223(E)

## Introduction and Problem Statement
A problem in Natural Language Processing (NLP) research is accumulating large training data corpus. Traditionally this involved costly methods such as using humans to manually categorize texts. However, with the rise of social media, we now have access to large amounts of semi-structured text. This allows for new methodologies for collecting and analyzing text. One such social media site, www.reddit.com, is especially useful because its members self segregate into groups (subreddits), which can act labels. In this project we empirically explore how best to use Reddit data as input to a classifier by trying different pre-processing steps and models to maximize classification accuracy.

The underlying idea behind this project is that the grammar, vocabulary, topics, and comment patterns are different across various subreddits based on the target demographics of the subreddits. With a large enough dataset, we can train a classifier to identify these variations in the comments and associate the comment patterns of the user to associate them to a subreddit and hence, that demographic. The scope of this project includes sets of subreddits that may be targeted to a specific, mutually exclusive demographics; for this we have initially selected $r/democrats$ and $r/republicans$. Once we have trained our model on these subreddits, we will attempt to infer information given a corpus of a user’s entire comment history.
