Since this dataset is artificial, any insight gained from here could be a false representation of reality. and we are going to see during this analysis the limits of what we can actually gain vs. what is clearly the propmt for the fabrication of the dataset. For example, if you plot Weight vs Height (for females and for males), you can clearly see that females were given all heights between 1.5 and 1.8m, and weights between 40 and 80kg. So this graph doesn't really provide useful information regarding the data. So some analysis will be excluded for this reasons. This is what I was able to find out about the dataset:

There are 511 males and 462 females.

By analyzing the correlation_Matrix.png, you can distinguish that several attributes are correlated (>abs(0.2) means correlation). For example:

Age, as expected, seems to have no distinguishable correlation with any other variable.

For gender I established that Male is -1 and Female is 1 so it can be used in the correlation matrix. Then being female seems to be correlated (correlation is positive) to high fat percentage (compared to males). On the other hand, being male seem to be correlated (correlation is negative) to a higher weight, height, water intake and BMI, compared to females.

