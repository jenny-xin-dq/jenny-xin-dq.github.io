---
title: "Amazon Echo Show Business Insights Report"
date: 2021-02-10
tags: [text analytics, NLP, sentiment analysis, data science]
header:
  image: "/images/SF-golden-gate.jpeg"
excerpt: "Text Analytics, NLP, Sentiment Analysis, Data Science"
mathjax: "true"
---

## Introduction

According to Statista, Amazon Echo’s unit sales was around 2.5 times more than that of Google
Home’s in 2018. However, in 2025, it is estimated that Google smart speakers will be selling 10
million more than Amazon Echo globally in 2025 (Statista, 2021). Facing increasing fierce
competition with Google Home, Amazon Echo needs to adjust its products and marketing
strategies accordingly to not only maintain, but also further improve its market share by 2025.
This report analyzes customer reviews from ebay on two similar products: the Amazon Echo
Show (2nd Gen) 10.1 " Smart Display with Alexa and Nest Hub Max 10" Smart Display with
Google Assistant. Both products are priced at the same level ($229.99) with similar size and
functions. The report presents findings on the two products by applying NLP algorithms on 40
most relevant reviews and 40 lowest rating reviews. Recommendations based on findings will
be presented at the end of the report.

## Methods and Findings
### Token Frequency Histogram and Word Cloud
Following the steps of tokenization, grouping, counting frequencies and removing stop-words,
we are able to generate token frequency histograms to see which words appear the most in the
corresponding reviews. Comparing most relevant reviews of Amazon Echo Show and Google
Nest Hub Max, we can see that
1. Amazon Echo Show buyers are more passionate towards the product by using the word
“love” much more frequently than Google Nest Hub Max users
2. Amazon Echo Show buyers commented more on the sound and music, while Google
Nest Hub Max buyers have more reflections on its camera and videos displayed
3. Amazon Echo Show buyers mentioned frequently on the word “easy”, however the word
doesn’t appear as frequently on Google Nest Hub Max buyers’ reviews

After installing and applying the “wordcloud” package, we are able to visualize the word
frequency more directly. From the word cloud generated for the lowest rating reviews, we
discover that Amazon Echo Show buyers mentioned more on the device’s “screen”, while it is
not a problem for Google Nest Hub Max buyers.

### Word Cloud with Sentiment
In order to further examine buyers’ emotions and intentions for the lowest rating reviews, we
apply nrc sentiment lexicon in the sentiment dataset in the tidytext package. The nrc sentiment
lexicon categorizes words in a binary fashion (yes or no) into categories of positive, negative,
anger, anticipation, disgust, fear, joy, sadness, surprise, and trust (Silge & Robinson, 2017).
From the word cloud inner joined with nrc sentiment lexicon, we find that among the lowest
rating reviews,
1. Google Nest Hub Max buyers are angrier than Amazon Echo Show buyers as they
complained more on the price and the quality of Nest Hub Max
2. Amazon Echo Show seems to be more often bought as gifts for special occasions
3. Amazon Echo Show buyers present more emotions of fear than Google Nest Hub Max
buyers

### Quadrogram Network
By applying the token = “ngrams” argument, we are able to explore relationships between words
in the customer reviews. We set n to 4 to examine pairs of four consecutive words, known as
“quadrograms''. From the following graphs below, we are able to see what words tend to be
adjacent to each other. There are similarities in the reviews as “weather” and “news” are
connected in both products, as well as “tv” and “lights”.

From Amazon Echo Show’s quadrogram network, we find that
1. For watching purposes, Amazon Echo Show buyers use the product to make calls,
watch TV and videos, check on weather and news
2. The connection in “coffee maker”, “tv” and “lights” impressed buyers
3. Connection in “family” and “fantastic” shows Amazon Echo Show brings amazing
experience to households
4. “Screen” is connected with “larger” and “bigger”, “sound” is connected with “surround”
5. “Voice” is connected with “smart” “controlled”
From Google Nest Hub Max’s quadrogram network, we find that
6. For watching purposes, Google Nest Hub Max users use it to watch “Youtube” and
“Netflix”
7. Google Nest Hub Max is more centered on the smart home system including controlling
lights, tv, garage, doors, locks, thermostat
8. Music service is connected with “excellent”

## Conclusions and Recommendations
Based on the findings on the most relevant reviews and lowest ranking reviews on both
products, we are able to identify Amazon Echo Show's characteristics of the customers,
customers’ experience and product’s competitive advantage. In order to increase Amazon Echo
Show’s market share and revenue for the upcoming years, following recommendations on
product development and marketing strategies are given accordingly:
1. The Amazon Echo Show users are passionate towards the product mostly based on the
smart, controlled voice of Alexa, music quality and its easiness to use. Echo Show
should continue its high quality on sound and convenient user interactions to further
expand its market by enhancing such brand image.
2. Since 1) Amazon Echo Show’s “screen” has been mentioned more frequently on its
lowest ranking reviews, 2) words such as “bigger” and “larger” are closely mentioned
with screen as well, product research and development should focus more on the screen
quality and size that most cater to customers’ needs and preferences
3. Amazon Echo Shows are often bought as gifts for special occasions. Therefore, Amazon
Echo can provide additional services such as free engravings, gift packaging, and
greeting notes to further improve customer experience. It should also consider festival
deals to boost its sales during the season.
4. The emotion of fear is generated more on Amazon Echo Show’s negative reviews. Echo
Show should create more safety and security options for users to select.
5. Amazon Echo Show has created fantastic family experiences for its users. In order to
reach out to more potential customers such as household owners, Amazon Echo Show
should 1) have its advertisements portraying more lovely family scenarios, 2) marketing
collaboration with furniture manufactures or IKEA store to attract the target market, 3) as
homeowners are mostly car owners, Amazon Echo Show should make partnerships with
electric/ autonomous driving cars to explore more opportunities in the promising field
By understanding its customers, advancing its product technology, improving market strategies
and foreseeing future opportunities, we are certain that Amazon Echo Show would be able to
increase its market share and lead the industry in the next decade.



