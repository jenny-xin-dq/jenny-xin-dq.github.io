---
title: "Unsupervised Analysis Project"
date: 2021-01-31
tags: [machine learning, unsupervised analysis, data science]
header:
  image: "/images/SF-golden-gate.jpeg"
excerpt: "machine learning, unsupervised analysis, data science"
mathjax: "true"
---

## Overview
Apple and Windows is a time-sized rivalry. It has always been a growing dilemma for a customer 
to make choices with each getting their own share of loyalists and critic, advantages and contrast. 
This question applies to producers who realize what makes that person purchase their goods and even 
more so who exactly purchases their goods. With demographic segmentation as a past, we plan to gather 
a detailed grasp of the purchasing behavior of customers who choose one over the other in the 
following studies. The three key drivers of purchasing are personal (demographic), psychological 
(psychographic or psychometric) and social (external stimuli).The research focuses on more intrinsic 
variables such as demographics and customer psychography. The collection of data is primary from 
an online survey. Considering that this is a consumer survey with limitations including lack of 
interest, misperception and dishonest responses at the end of a long survey. Psychology helping 
us understand the motives, actions, emotions and cognition of an individual. We can look at 
demographics gathered while reviewing the data collection. The psychographics of consumers are 
effectively known. The survey used Big 5 as well as Hult DNA, to determine the behavior.

###Big 5
The Big 5 Personality, also called OCEAN, allows us to understand the degree to which customers use:
1. Openness
2. Conscientousness
3. Extraversion
4. Agreeableness
5. Neuroticism
This will help us understand the behaviour of the student.

###Hult DNA
For a student to be employable in graduation, nine years of study at the Hult facilities have 
resulted in a must-do approach. Within its students Hult frequently absorbs them.The Hult DNA 
consists of 3 broad categories and 3 subcategories within each category, namely:

1. Thinking

Shows self-awareness
Embraces change
Demonstrates dynamic thinking

2.Communicating

Speaks and listens confidentally
Influences confidentally
Presents ideas effectively

3.Team Building

Fosters collaborative relationships
Influences productivity
Resolves conflict constructively
This helps us to understand the student's psychometrics.

##Dataset Analysis
There are 137 students in the survey dataset, 78 questions based on psychological and 7 questions 
on demographic. 50 questions related to Big5 and 18 questions related to Hult-DNA. 69 students 
have Macbook and 68 students have Windows. 77 students want Macbook and 68 students want Windows 
next year and 4 students to Chromebook. Total of 23 students want to change their laptop from 
which 17 students moving from windows to macbook and only 6 students from macbook to windows. 
Out of 78 male students, 15 of them are changing their laptop and 8 female students out of 59 
want to change their laptop.

**Importing Packages**

```python
import sys                      # system-specific parameters and functions
import pandas as pd # data science essentials
import seaborn as sns# essential graphical output
import matplotlib.pyplot as plt # enhanced graphical output
import numpy as np # mathematical essentials
import statsmodels.formula.api as smf # regression modeling
from sklearn.decomposition import PCA            # pca
from sklearn.datasets      import load_digits    # digits dataset
from sklearn.manifold      import TSNE           # t-SNE
from sklearn.preprocessing import StandardScaler # standard scaler
import matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms
from sklearn.cluster         import KMeans     

# reading the file into Python
team_df = pd.read_excel('./survey_data.xlsx')

#lowering case of the column names
#team_df.columns = map(str.lower, team_df.columns)

########################################
# loading data and setting display options
########################################
# loading data
digits = load_digits()

# setting print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
pd.np.set_printoptions(threshold=sys.maxsize)

########################################
# chacking the type of the dataset
########################################
type(digits)
```

**Changing column names**
We will start with data cleaning and creation of new variables for our analysis

```python
#Creating a new list with the new names for each column
col_dict = {
    "surveyID": "survey_ID_num",
    "Am the life of the party": "life_of_party",
    "Feel little concern for others": "concern_others",
    "Am always prepared": "always_prepared",
    "Get stressed out easily": "stressed_easily",
    "Start conversations":"start_conversations",
    "Have a rich vocabulary": "rich_vocabulary",
    "Don't talk a lot": "do_not_talk",
    "Am interested in people": "int_people",
    "Leave my belongings around":"leave_belong_around",
    "Am relaxed most of the time":"relax_most_time",
    "Have difficulty understanding abstract ideas":"diff_und_abstract",
    "Feel comfortable around people":"conf_with_people",
    "Insult people":"insult_people",
    "Pay attention to details":"attention_details",
    "Worry about things":"worry_things",
    "Have a vivid imagination":"vivind_immag",
    "Keep in the background":"keep_background",
    "Sympathize with others' feelings":"symp_others",
    "Make a mess of things":"make_mess",
    "Seldom feel blue":"seldon_feel_blue",
    "Am not interested in abstract ideas":"uninterested_abstract_ideas",
    "Am not interested in other people's problems":"uncurious_about_people_problems",
    "Get chores done right away":"chores_done_right_away",
    "Am easily disturbed":"easily_disturbed",
    "Have excellent ideas":"have_excellent_ideas",
    "Have little to say":"have_little_to_say",
    "Have a soft heart":"soft_hearted",
    "Often forget to put things back in their proper place":"forget_to_place_back_inorder",
    "Get upset easily":"get_upset_easily",
    "Do not have a good imagination":"bad_imagination",
    "Talk to a lot of different people at parties":"social_at_parties",
    "Am not really interested in others":"not_intrested_in_others",
    "Like order":"like_order",
    "Change my mood a lot":"frequent_mood_change",
    "Am quick to understand things":"fast_learner",
    "Don't like to draw attention to myself":"reticent_person",
    "Take time out for others":"gives_time_for_others",
    "Shirk my duties":"shrik_my_duties",
    "Have frequent mood swings":"frequent_mood_swings",
    "Use difficult words":"use_difficult_words",
    "Don't mind being the center of attention":"center_of_attention",
    "Feel others' emotions":"feel_others_emotions",
    "Follow a schedule":"follows_schedule",
    "Get irritated easily":"mad_easily",
    "Spend time reflecting on things":"spend_time_reflecting",
    "Am quiet around strangers": "quiet_with_strangers",
    "Make people feel at ease":"make_people_feel_ease",
    "Am exacting in my work":"exact_in_work",
    "Often feel blue":"often_feel_blue",
    "Am full of ideas":"full_of_ideas",
    "See underlying patterns in complex situations":"answers_complex_situations",
    "Don't  generate ideas that are new and different":"dont_create_new_ideas",
    "Demonstrate an awareness of personal strengths and limitations":"self_awareness",
    "Display a growth mindset":"growth_mindset",
    "Respond effectively to multiple priorities":"respond_effectively_1",
    "Take initiative even when circumstances, objectives, or rules aren't clear":"takes_initiative_1",
    "Encourage direct and open discussions":"encourage_open_discussions_1",
    "Respond effectively to multiple priorities.1": "respond_effectively_2",
    "Take initiative even when circumstances, objectives, or rules aren't clear.1":"takes_initiative_2",
    "Encourage direct and open discussions.1":"encourage_open_discussions_2",
    "Listen carefully to others":"listen_others",
    "Don't persuasively sell a vision or idea":"dont_sell_idea",
    "Build cooperative relationships": "build_coop_rel",
    "Work well with people from diverse cultural backgrounds": "work_diverse_cult",
    "Effectively negotiate interests, resources, and roles": "effect_negotiate",
    "Can't rally people on the team around a common goal": "cant_rally",
    "Translate ideas into plans that are organized and realistic": "translate_ideas_to_plans",
    "Resolve conflicts constructively": "resolve_conflicts",
    "Seek and use feedback from teammates": "seek_use_feedback",
    "Coach teammates for performance and growth": "coach_for_perf_growth",
    "Drive for results": "drive_for_results",
    "What laptop do you currently have?": "current_laptop",
    "What laptop would you buy in next assuming if all laptops cost the same?": "next_laptop",
    "What program are you in?": "program",
    "What is your age?": "age",
    "Gender": "gender",
    "What is your nationality? ": "nationality",
    "What is your ethnicity?": "ethnicity"
}
#Assigning the names to the columns
team_df.columns = team_df.columns.map(col_dict)

```

**Merging similar columns**

```python
#Creating new columns by adding the ones that were repeated in the dataset
team_df['respond_effectively_3']  = (team_df['respond_effectively_1']+team_df['respond_effectively_2'])/2
team_df['takes_initiative_3']  = (team_df['takes_initiative_1']+team_df['takes_initiative_2'])/2
team_df['encourage_open_discussions_3']  = (team_df['encourage_open_discussions_1']+team_df['encourage_open_discussions_2'])/2
```
```python
#Creating a loop to assign these values to the column respond effectively
team_df['respond_effectively']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'respond_effectively_3'] >= 0 and team_df.loc[index, 'respond_effectively_3'] < 1:
        team_df.loc[index, 'respond_effectively'] = 0
        
    if team_df.loc[index, 'respond_effectively_3'] >= 1 and team_df.loc[index, 'respond_effectively_3'] < 2:
        team_df.loc[index, 'respond_effectively'] = 1

    if team_df.loc[index, 'respond_effectively_3'] >= 2 and team_df.loc[index, 'respond_effectively_3'] < 3:
        team_df.loc[index, 'respond_effectively'] = 2
    
    if team_df.loc[index, 'respond_effectively_3'] >= 3 and team_df.loc[index, 'respond_effectively_3'] < 4:
        team_df.loc[index, 'respond_effectively'] = 3
        
    if team_df.loc[index, 'respond_effectively_3'] >= 4 and team_df.loc[index, 'respond_effectively_3'] < 5:
        team_df.loc[index, 'respond_effectively'] = 4
        
    if team_df.loc[index, 'respond_effectively_3'] == 5:
        team_df.loc[index, 'respond_effectively'] = 5
```
```python
#Creating a loop to assign these values to the column takes initiative
team_df['takes_initiative']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'takes_initiative_3'] >= 0 and team_df.loc[index, 'takes_initiative_3'] < 1:
        team_df.loc[index, 'takes_initiative'] = 0
        
    if team_df.loc[index, 'takes_initiative_3'] >= 1 and team_df.loc[index, 'takes_initiative_3'] < 2:
        team_df.loc[index, 'takes_initiative'] = 1

    if team_df.loc[index, 'takes_initiative_3'] >= 2 and team_df.loc[index, 'takes_initiative_3'] < 3:
        team_df.loc[index, 'takes_initiative'] = 2
    
    if team_df.loc[index, 'takes_initiative_3'] >= 3 and team_df.loc[index, 'takes_initiative_3'] < 4:
        team_df.loc[index, 'takes_initiative'] = 3
        
    if team_df.loc[index, 'takes_initiative_3'] >= 4 and team_df.loc[index, 'takes_initiative_3'] < 5:
        team_df.loc[index, 'takes_initiative'] = 4
        
    if team_df.loc[index, 'takes_initiative_3'] == 5:
        team_df.loc[index, 'takes_initiative'] = 5
```
```python
#Creating a loop to assign these values to the column encourage open discussions
team_df['encourage_open_discussions']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'encourage_open_discussions_3'] >= 0 and team_df.loc[index, 'encourage_open_discussions_3'] < 1:
        team_df.loc[index, 'encourage_open_discussions'] = 0
        
    if team_df.loc[index, 'encourage_open_discussions_3'] >= 1 and team_df.loc[index, 'encourage_open_discussions_3'] < 2:
        team_df.loc[index, 'encourage_open_discussions'] = 1

    if team_df.loc[index, 'encourage_open_discussions_3'] >= 2 and team_df.loc[index, 'encourage_open_discussions_3'] < 3:
        team_df.loc[index, 'encourage_open_discussions'] = 2
    
    if team_df.loc[index, 'encourage_open_discussions_3'] >= 3 and team_df.loc[index, 'encourage_open_discussions_3'] < 4:
        team_df.loc[index, 'encourage_open_discussions'] = 3
        
    if team_df.loc[index, 'encourage_open_discussions_3'] >= 4 and team_df.loc[index, 'encourage_open_discussions_3'] < 5:
        team_df.loc[index, 'encourage_open_discussions'] = 4
        
    if team_df.loc[index, 'encourage_open_discussions_3'] == 5:
        team_df.loc[index, 'encourage_open_discussions'] = 5
```
```python
#Creating a loop to group age into a corresponding category
team_df['age_group']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'age'] >= 21 and team_df.loc[index, 'age'] <= 29:
        team_df.loc[index, 'age_group'] = 'twenties'
        
    if team_df.loc[index, 'age'] >= 30 and team_df.loc[index, 'age'] <= 39:
        team_df.loc[index, 'age_group'] = 'thirties'

    if team_df.loc[index, 'age'] >= 40 and team_df.loc[index, 'age'] <= 49:
        team_df.loc[index, 'age_group'] = 'forties'
```
```python
#Creating a list to clean the nationalities and have them in the same format
group_nationalities = {"Indian" : "Indian",
                       "China": "Chinese",
                       "German": "German",
                       "Mexican": "Mexican",
                      "Peruvian": "Peruvian",
                      "Taiwan": "Taiwanese",
                      "American": "American",
                      "Chinese": "Chinese",
                      "USA" : "American",
                      "Brazilian" : "Brazilian",
                      "Norwegian" : "Norwegian",
                      "Russian" : "Russian",
                      "Colombian" : "Colombian",
                      "Turkish" : "Turkish",
                      "Nigerian" : "Nigerian",
                      "Vietnamese" : "Vietnamese",
                      "Republic of Korea" : "South Korean",
                      "Indonesian" : "Indonesian",
                      "Italian" : "Italian",
                      "Thai": "Thai",
                      "Russia" : "Russian",
                      "indian" : "Indian",
                      "Brazil" : "Brazilian",
                      "British" : "British",
                      "Mauritius" : "Mauritian",
                      "chinese" : "Chinese",
                      "colombian" : "Colombian",
                      "German/American" : "Multi-ethnic",
                      "Costarrican" : "Costarrican",
                      "Nigeria" : "Nigerian",
                      "Germany" : "German",
                      "Japan" : "Japanese",
                      "Czech" : "Czech",
                      "mexican" : "Mexican",
                      "canadian" : "Canadian",
                      "Kenyan" : "Kenyan",
                      "Ghanaian" : "Ghanaian",
                      "Belgian " : "Belgian",
                      "INDIAN" : "Indian",
                      "Indonesia" : "Indonesian",
                      "Philippines" : "Filipino",
                      "Ecuador" : "Ecuadorian",
                      "Ugandan" : "Ugandan",
                      "Korea" : "South Korean",
                      "Spain": "Spanish",
                      "Canada" : "Canadian",
                      "Italian and Spanish" : "Multi-ethnic",
                      "South Korea" : "South Korean",
                      "prefer not to answer" : "Prefer not to answer",
                      "china": "Chinese",
                      "peru": "Peruvian",
                      "Swiss": "Swiss",
                      "Portuguese" : "Portuguese",
                      "Belarus": "Belarusians",
                      "Ukrainian": "Ukrainian",
                      "ecuador" : "Ecuadorian",
                      "Dominican " : "Dominican",
                      "Congolese" : "Congolese",
                      "nigerian": "Nigerian",
                      "Pakistani": "Pakistani",
                      "Ecuadorian" : "Ecuadorian",
                      "italian": "Italian",
                      "Dominican" : "Dominican",
                      "indian." : "Indian",
                      "Venezuelan": "Venezuelan",
                      "CHINA" : "Chinese",
                      "British, Indian" : "Multi-ethnic",
                      "Kyrgyz" : "Kyrgyz",
                      "Spanish" : "Spanish",
                      "Panama" : "Panamanians",
                      "Colombia" : "Colombian",
                      "Filipino " : "Filipino",
                      "Congolese (DR CONGO)" : "Congolese",
                      "Czech Republic" : "Czech",
                      "Peru" : "Peruvian"}

#Assigning the names to the nationality column 
team_df['nationality'].replace(group_nationalities, inplace = True)

team_df['nationality'].value_counts()
```
```python
#Creating a list to group the nationalities into different countries
mapping = {"Indian": "Asia", 
           "Chinese": "Asia", 
           "Taiwanese": "Asia",
       "Vietnamese": "Asia",
           "South Korean": "Asia",
           "Indonesian": "Asia",
       "Thai": "Asia",
           "Japanese": "Asia",
           "Pakistani": "Asia",
           "Kyrgyz": "Asia", 
           "Filipino": "Asia",
           "Filipino1": "Asia",
           "Mauritian" : "Africa", 
           "Nigerian": "Africa", 
           "Kenyan": "Africa", 
           "Ghanaian": "Africa", 
           "Congolese": "Africa", 
           "Ugandan": "Africa",
           "Mexican": "North_America", 
           "American": "North_America", 
           "Canadian": "North_America",
           "Dominican": "North_America", 
           "Costarrican": "North_America", 
           "Panamanians": "North_America",
           "Dominican1": "North_America",
           "Peruvian": "South_America", 
           "Brazilian": "South_America",
           "Colombian": "South_America", 
           "Ecuadorian": "South_America", 
           "Venezuelan": "South_America",
           "German": "Europe", 
           "Norwegian": "Europe",
           "Russian": "Europe",
           "Turkish": "Europe",
           "Italian": "Europe",
           "British": "Europe", 
           "Czech": "Europe", 
           "Belgian": "Europe", 
           "Portuguese": "Europe",
           "Spanish": "Europe", 
           "Swiss": "Europe", 
           "Belarusians": "Europe",
           "Ukrainian": "Europe",
           "Multi-ethnic": "Others", 
           "Prefer not to answer": "Others"}

#Assigning the nationalities to the new column created
team_df['country_mapped'] = team_df.nationality.map(mapping)
```
```python
#Creating new variables by making a comparison of people that can switch laptop model
team_df['change_laptop']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Macbook' :
        team_df.loc[index, 'change_laptop'] = 'no_change'
    
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Windows laptop' :
        team_df.loc[index, 'change_laptop'] = 'change_of_laptop'
        
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Chromebook' :
        team_df.loc[index, 'change_laptop'] = 'change_of_laptop'
        
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Windows laptop' :
        team_df.loc[index, 'change_laptop'] = 'no_change'
    
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Macbook':
        team_df.loc[index, 'change_laptop'] = 'change_of_laptop'
        
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Chromebook' :
        team_df.loc[index, 'change_laptop'] = 'change_of_laptop'
```
```python
#Creating a new variable to see Apples loyalty customers
team_df['apple_target']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Macbook' :
        team_df.loc[index, 'apple_target'] = 'customer_genuine'
    
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Windows laptop' :
        team_df.loc[index, 'apple_target'] = 'customer_loss'
        
    if team_df.loc[index, 'current_laptop'] == 'Macbook' and team_df.loc[index, 'next_laptop'] == 'Chromebook' :
        team_df.loc[index, 'apple_target'] = 'customer_loss'
        
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Windows laptop' :
        team_df.loc[index, 'apple_target'] = 'target_customer'
    
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Macbook':
        team_df.loc[index, 'apple_target'] = 'customer_gain'
        
    if team_df.loc[index, 'current_laptop'] == 'Windows laptop' and team_df.loc[index, 'next_laptop'] == 'Chromebook' :
        team_df.loc[index, 'apple_target'] = 'target_customer'
```
```python
#Creating a new variable to compare the change of laptop and gender
team_df['change_gender']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'gender'] == 'Male' :
        team_df.loc[index, 'change_gender'] = 1
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'gender'] == 'Female' :
        team_df.loc[index, 'change_gender'] = 2
    
    if team_df.loc[index, 'change_laptop'] == 'no_change':
        team_df.loc[index, 'change_gender'] = 0
```
```python
#Creating a new variable to compare the change of laptop and age
team_df['change_age']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'age_group'] == 'twenties' :
        team_df.loc[index, 'change_age'] = 1
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'age_group'] == 'thirties' :
        team_df.loc[index, 'change_age'] = 2
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'age_group'] == 'forties' :
        team_df.loc[index, 'change_age'] = 3
    
    if team_df.loc[index, 'change_laptop'] == 'no_change':
        team_df.loc[index, 'change_age'] = 0
```
```python
#Creating a new variable to compare the change of laptop and the program
team_df['change_degree']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MIB & Business Analytics)' :
        team_df.loc[index, 'change_degree'] = 1
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'One year Business Analytics' :
        team_df.loc[index, 'change_degree'] = 2
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MBA & Business Analytics)':
        team_df.loc[index, 'change_degree'] = 3
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MBA & Disruptive innovation)':
        team_df.loc[index, 'change_degree'] = 4
        
    if team_df.loc[index, 'change_laptop'] == 'no_change':
        team_df.loc[index, 'change_degree'] = 0
 ```
 **Dropping columns not required**
 ```python
 #Creating a new variable to compare the change of laptop and the program
team_df['change_degree']   = 0

for index, value in team_df.iterrows():
    
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MIB & Business Analytics)' :
        team_df.loc[index, 'change_degree'] = 1
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'One year Business Analytics' :
        team_df.loc[index, 'change_degree'] = 2
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MBA & Business Analytics)':
        team_df.loc[index, 'change_degree'] = 3
    
    if team_df.loc[index, 'change_laptop'] == 'change_of_laptop' and team_df.loc[index, 'program'] == 'DD (MBA & Disruptive innovation)':
        team_df.loc[index, 'change_degree'] = 4
        
    if team_df.loc[index, 'change_laptop'] == 'no_change':
        team_df.loc[index, 'change_degree'] = 0
```
```python
#lowering case of the column names
team_df.columns = map(str.lower, team_df.columns)

# checking information about each column
team_df.head(5)
```
        
