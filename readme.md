# About this project

During my PhD I measured the current-density vs. voltage curves for a wide variety of solar cells. For this project I am trying to look at handling some of that data, fitting these data with a model, and extracting performance parameters of the cell.

At the top of my example datafile there are several lines that show the calculated parameters based on a LabView program in our lab. My goal will be to fit the data myself and match my own calculated parameters to this header. 

First, I tried using the ideal diode equation to fit my data:

$$J(V) = J_{D} (exp(A V) - 1) - J_{sc}$$

An example of the fitted data can be seen below:

![fitted_JV](JVfit.png)

As seen above, the fitted curve is less square-shaped than the raw data - meaning that the fit will underestimate the fill factor parameter of the solar cell. This issue will also end up under-estimating the open circuit voltage of the device as well. 

The predicted parameters are as follows:

![solar_characteristics]('chars.png')

If I take a ratio of these values with the expected parameters from the heading of my datafile, (calculated/expected), I get the following values for the three devices:

![solar_characteristics_ratios]('char_ratio.png')

This shows how the model over-predicted the Jsc, but under-predicted the FF and Voc. This led to the overall efficiency to be lower than the expected value. 