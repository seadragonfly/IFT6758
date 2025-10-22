# Université de Montréal
# IFT-6758-B  -  A22  -  Data Science 

# Assignment 3

Grading breakdown:

| Section                           | Required files        | Score |
|-----------------------------------|-----------------------|:--:|
| Reddit Weekends                   | `reddit_weekends.py`  | 35 |
| &emsp;+ graphics, short answers   | `reddit_weekends.ipynb` | 10 |
| Chess Ratings                     | `chess_ratings.py`    | 35 |
| &emsp;+ graphics, short answers   | `chess_ratings.ipynb` | 20 |


In general, your assignments will be self-graded, meaning you **must not change the signature of the defined functions** (same inputs and outputs).
For questions involving Numpy and Pandas, you **must** handle all iteration via calls to the relevant library functions.
This means you cannot use native Python for loops, list comprehensions, etc., when something could have been done with a native pandas/numpy call.
Loops are otherwise allowed.


### Submission

To submit files, please submit **only the required files** (listed in the table above) that you have completed in **GradeScope**.
Do not include data or other miscellaneous files.
You do not need to submit all files at once to run the autograder.
For example, if you have only completed `reddit_weekends.py`, you do not need to submit the rest of the files to get feedback on `reddit_weekends.py`.


## 1. Week-ends Reddit

This question uses data derived from the Reddit Comment Archive, which is a collection of every Reddit comment, distributed as 150 GB of compressed JSON.

The provided file `reddit-counts.json.gz` contains a count of the number of comments posted daily in each Canadian province subreddit and in r/canada itself.
Again, the format is line-by-line compressed JSON. It turns out that Pandas (≥0.21) can handle compression automatically, and we don't need to explicitly decompress:

``
```bash
counts = pd.read_json(path, lines=True)
```
We ask ourselves the following question: Are there different numbers of Reddit comments posted on weekdays than on weekends?

For this question, we will only examine the values ​​in 2012 and 2013 and in the r/canada subreddit. First, create a DataFrame with the provided data and separate weekdays from weekends.

Hint : use `datetime.date.weekday`. Values ​​equal to 5 are Saturday and 6 are Sunday.

Complete the `reddit_weekends.py` program for this question and follow the `reddit_weekends.ipynb` notebook.
You will receive marks for your work in the Python file, as well as for the numbers and short answers in the notebook.
The Python file also has a section of code to execute. You can run it via the command line with the following command:

```bash
python3 reddit_weekends.py ./data/reddit-counts.json.gz
```

Note that the output produced by the above command will not be marked. We will test your functions directly via unit tests.
### 1.1 Student’s T-Test

- Complete `reddit_weekends.py:process_data()`
- Complete `reddit_weekends.py:tests()`

Use `scipy.stats` to perform a T-test on the data to obtain a p-value.  Can you conclude that there is a different number of Reddit comments posted on weekdays compared to weekends?  Try `stats.normaltest` to see if the data are normally distributed, and `stats.levene` to see if the two datasets have equal variances. Do you think you can draw a conclusion now? (Hint: no, we get “0.0438”.)


### 1.2 Solution 1: Data Transformation

- Complete the plots in `reddit_weekends.ipynb`

Take a look at the histogram of the data. You will notice it is skewed — that’s why it wasn’t normally distributed in the previous section. Transform the counts so that the data do not fail the normality test.  Possible transformations: `np.log`, `np.exp`, `np.sqrt`, `counts**2`.  Choose the one that gets closest to a normal distribution. (Hint: none of them will pass the normality test.)


### 1.3 Solution 2: Central Limit Theorem

- Complete `reddit_weekends.py:central_limit_theorem()`
- Complete the plots in `reddit_weekends.ipynb`

The Central Limit Theorem states that if our sample size is large enough and we look at the sample means, the result should be normally distributed.  We will combine all weekdays and weekends for each year/week pair and take the mean of their (non-transformed) counts.

Hint: you can get a “year” and “week number” from the first two values returned by `date.isocalendar()`.  This year and week number will give you a unique identifier for each week.  Use Pandas to group by this value and aggregate by taking the mean.  
However, note that the year returned by `isocalendar` is not always the same as the date’s actual year — use the `isocalendar` year, which is correct in this case.

Check these values to see if they have a Gaussian distribution and equal variance. Apply a T-test if it makes sense.

We should note that we’re subtly changing the question here.  It’s now something like: “Does the number of comments per week differ between weekdays and weekends?”


### 1.4 Solution 3: Non-Parametric Test

- Complete `reddit_weekends.py:mann_whitney_u_test()`

Another option is a statistical test that does not rely as heavily on the shape of its input.  The Mann-Whitney U test does not assume normal distribution or equal variances between groups. 

Perform a U-test on the (original, non-transformed, non-aggregated) counts. Note that it should be a two-sided test — make sure the function arguments are correct.

Note that we are again subtly changing the question. If we reach a conclusion based on a U-test, it’s something like:  “It is not equally likely that there are more comments on weekends compared to weekdays.”


### 1.5 Questions

Answer these questions in the specified section of `reddit_weekends.ipynb`:

- Which of the four suggested transformations gets you closest to satisfying the assumptions of a T-test?
- We’ve given rough verbal explanations of what the weekly test and the Mann-Whitney test were actually testing.  
  Do the same for the original T-test and the T-test on transformed data.  
  In other words, describe what the conclusion would be if you could reject the null hypothesis for those tests.
- Among the four approaches, which one do you think best answers the initial question:  
  “Is there a different number of Reddit comments posted during the week and on the weekend?”  
  Briefly explain why. (There’s not necessarily a single correct answer, but there are wrong ones!)
- On average, are there more Reddit comments posted in r/canada on weekdays or weekends?


## 2. Chess Rankings

When comparing the performance between two groups, a common mistake is to compare only the best of each group — especially in cases where one group has significantly more samples than the other (overrepresented group). A classic example is chess, where people have incorrectly claimed that women are worse players than men because no woman has ever been world champion, or because the Elo rating (a method for calculating relative skill levels) of the top woman is much lower than that of the top man.

Why is this wrong? Because it ignores participation rates between groups.   Naively, if two groups are drawn from the same distribution, the overrepresented group will have more chances to appear in the distribution’s tails. For example, suppose there are two groups of people: A and B. Group A has 10 people, and group B has 2. Each of the 12 people is randomly assigned a number between 1 and 100 (with replacement). Then, take the maximum of group A as its score, and the maximum of group B as its score. On average group A will score around 91, and group B around 67 — even though the only difference is sample size. The larger group has a higher chance of achieving a high score, and thus a higher average maximum. The fair way to compare groups of unequal size is to compare their *means*, not their *maximums*. In this case, both groups would have an average of 50 — no difference! This only highlights the statistical effect of comparing overrepresented vs underrepresented groups, and ignores any real-world or data-related biases.

In this section, we will explore this concept in the context of chess rankings. No starter code is provided in the Python file; instead, follow along in the `chess_ratings.ipynb` notebook.


### 2.1 Load and Clean the Data

- Complete `chess_ratings.py:parse_xml()`
- Complete `chess_ratings.py:clean_data()`

A zipped XML file is provided:

    ./data/standard_oct22frl_xml.zip

It was obtained from the FIDE player database, available [here](http://ratings.fide.com/download.phtml?period=2022-10-01). Note that ratings are updated regularly, so we use the fixed timestamp of October 6, 2022, provided in the data directory. You should not need to download anything.

Complete the `parse_xml()` method. The first part of the method already includes code to automatically extract the zip file for you. Make sure to extract the following fields from the XML (and use them as column names) in your DataFrame:

    ["name", "rating", "sex", "birthday", "country", "flag", "title"]

Then complete `clean_data()`: Remove players with NaN birthdays, convert numeric types appropriately, and filter for birthdays `<=2002` (since Elo scores for players under 20 may not be reliable).


### 2.2 Compare Rating Distributions

- Complete `chess_ratings.py:bin_counts()`
- Complete the plots in `chess_ratings.ipynb`

Now compare the rating distributions for male and female players. Since the data are quite granular, we’ll need to bin the ratings. Complete `bin_counts()`, which should handle arbitrary data and bin selection. In addition to returning raw counts, also return normalized counts (`count_norm`).

You’ll use this method to complete the plots in `chess_ratings.ipynb`. For the plots, choose a *bin width* of 50, a *range* from 1000 to 2900 (make sure your range extends to 2900!), and the *center* of each bin as its midpoint (e.g., for a bin [1500, 1550], the center is 1525). Note that when defining bins, you’ll find `len(bin_centers) = len(bins) - 1`.

Using the binned data, create two side-by-side line plots: one for raw counts (`count`) and one for normalized counts (`count_norm`). M/F should appear in two different colors. Be sure to include these plots in your notebook when submitting.


### 2.3 Permutation Tests

- Complete `chess_ratings.py:PermutationTests.job()`
- Complete `chess_ratings.py:sample_two_groups()`

Finally, we will perform the permutation tests described in the introduction. Take the complete cleaned dataset (men and women) and randomly sample two groups *without replacement*. The group sizes should reflect the real-world imbalance (male vs female population sizes). Complete `PermutationTests.job()`, which implements the sampling part of this experiment, and return the *maximum values* of the overrepresented and underrepresented groups respectively.

Then complete `sample_two_groups()`, which runs this experiment `n_iter` times. Once done, run this experiment in the notebook with at least `n_iter=1000`. Execute the cell that prints the mean difference obtained from the permutation tests, as well as the actual observed difference.  Be sure to include these printed results in your notebook submission.


### 2.4 Questions

Answer these questions (1–3 sentences each) in the specified section of `chess_ratings.ipynb`:

1. Interpret the results — can you draw a conclusion? Remember that the original claim was: “Men are better than women at chess because most of the top players are men.” (*Note: part of your answer here will likely connect to the next question.*)
2. Do you think the numbers obtained here tell the whole story? What might be the problems with this analysis?  Are the data biased in any way (other than overrepresentation bias)? Is Elo a good measure, and can it really answer the original question?  Are there social, cultural, or systemic factors that might prevent the underrepresented group from achieving similar results?

The purpose of these questions is to emphasize that data represent only a limited view of the real world.  As data scientists, it’s crucial to step back and reflect on how our findings relate to reality — rather than naively assuming that the data and experimental setup are perfect. Otherwise, we risk drawing false or misleading conclusions.  There may be multiple causal factors behind an observed relationship, independent of the original hypothesis — for example:  using data that don’t truly reflect the question being tested, biased data (including overrepresentation), or genuine systemic differences between groups.

# References

- **Reddit Weekends** is based on Greg Baker’s Data Science course at SFU.  
- **Chess Ratings** is based on the analysis in this [blog post.](https://en.chessbase.com/post/what-gender-gap-in-chess)
