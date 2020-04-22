#!/usr/bin/env python
# coding: utf-8

# In[20]:


#from datetime import datetime
#startTime = datetime.now()


# 
# ***
# 
# 
# <br><h2>A2.- Final Model      |       Case: Apprentice Chef, Inc.</h2>
# <h4>MSBA5 - Valencia           |       Machine Learning</h4>
# Jorge Hernández Jiménez - Marketing Analyst<br>
# Hult International Business School<br><br><br>
# 
# 
# 
# ***
# 

# In[21]:


################################################################################
# Import Packages & Load Defined Functions
################################################################################

# importing libraries
import pandas                   as   pd      # data science essentials
import matplotlib.pyplot        as   plt     # essential graphical output
import seaborn                  as   sns     # enhanced graphical output
#import statsmodels.formula.api  as   smf     # regression modeling
#import sklearn.linear_model                  # (scikit-learn)linear models (LinearRegression, Ridge, Lasso, ARD)
#import random                   as   rand
#import pydotplus                             # interprets dot objects

from sklearn.model_selection    import train_test_split             # train-test split
#from sklearn.preprocessing      import StandardScaler              # standard scaler
#from sklearn.metrics            import confusion_matrix            # confusion matrix
from sklearn.metrics            import roc_auc_score                # auc score
#from sklearn.neighbors          import KNeighborsClassifier        # KNN for classification
#from sklearn.neighbors          import KNeighborsRegressor         # KNN for Regression
#from sklearn.linear_model       import LogisticRegression          # logistic regression
from sklearn.tree               import DecisionTreeClassifier       # classification trees
#from sklearn.tree               import export_graphviz             # exports graphics
#from sklearn.externals.six      import StringIO                    # saves objects in memory
#from IPython.display            import Image                       # displays on frontend
#from sklearn.model_selection    import GridSearchCV                # hyperparameter tuning
#from sklearn.metrics            import make_scorer                 # customizable scorer
#from sklearn.ensemble           import RandomForestClassifier      # random forest
#from sklearn.ensemble           import GradientBoostingClassifier  # gbm



################################################################################
# Load Data
################################################################################

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'


# reading the file into Python
apprentice = pd.read_excel(file)



################################################################################
# Feature Engineering 
################################################################################

#We are going to create a column for email domain
# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in apprentice.iterrows():
    
    # splitting email domain at '@'
    split_email = apprentice.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# displaying the results
#email_df




# STEP 2: concatenating with original DataFrame

# safety measure in case of multiple concatenations
#apprentice = pd.read_excel('Apprentice_Chef_Dataset.xlsx')

# renaming column to concatenate
email_df.columns = ['NAME' , 'EMAIL_DOMAIN']


# concatenating email_domain with friends DataFrame
apprentice = pd.concat([apprentice, email_df['EMAIL_DOMAIN']],
                   axis = 1)


# printing value counts of personal_email_domain
#apprentice.loc[: ,'EMAIL_DOMAIN'].value_counts()




# STEP 3

#We can create new groups : personal and job emails

# email domain types
PERSONAL_EMAIL_DOMAINS = ['@gmail.com', '@protonmail.com', '@yahoo.com']

JUNK_EMAIL_DOMAINS = ['@me.com', '@aol.com', '@hotmail.com', '@live.com', 
                      '@msn.com', '@passport.com']


# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in apprentice['EMAIL_DOMAIN']:
        if   '@' + domain in PERSONAL_EMAIL_DOMAINS:
             placeholder_lst.append('personal')
            
        elif '@' + domain in JUNK_EMAIL_DOMAINS:
             placeholder_lst.append('junk')
            
            # I did not create a list for jobs, for new clients, there is much more variety of professional emails
            # personal emails are mainly the ones that are in that list, plus a few others we should add, but not
            # necessary for this excercise
        else:
            placeholder_lst.append('job')


# concatenating with original DataFrame
apprentice['DOMAIN_GROUP'] = pd.Series(placeholder_lst)


# checking results
#apprentice['DOMAIN_GROUP'].value_counts()





#Creating outlier thresholds

AVG_TIME_PER_SITE_VISIT_HI        =     200
AVG_PREP_VID_TIME_LOW             =     50
AVG_PREP_VID_TIME_HI              =     200
AVG_CLICKS_PER_VISIT_LOW          =     10
AVG_CLICKS_PER_VISIT_HI           =     16
TOTAL_MEALS_ORDERED_HI            =     150
UNIQUE_MEALS_PURCH_HI             =     6
CONTACTS_W_CUSTOMER_SERVICE_LOW   =     4
CONTACTS_W_CUSTOMER_SERVICE_HI    =     9
CANCELLATIONS_BEFORE_NOON_HI      =     2
CANCELLATIONS_AFTER_NOON_HI       =     2
PC_LOGINS_LOW                     =     4
PC_LOGINS_HI                      =     7
MOBILE_LOGINS_LOW                 =     0.5
MOBILE_LOGINS_HI                  =     3
WEEKLY_PLAN_HI                    =     5
EARLY_DELIVERIES_HI               =     2
LATE_DELIVERIES_HI                =     3
FOLLOWED_RECOMMENDATIONS_PCT_HI   =     40
LARGEST_ORDER_SIZE_LOW            =     3
LARGEST_ORDER_SIZE_HI             =     6
MASTER_CLASSES_ATTENDED_HI        =     2
TOTAL_PHOTOS_VIEWED_HI            =     300



##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers

# AVG_TIME_PER_SITE_VISIT
apprentice['OUT_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = apprentice.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][apprentice['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HI]

apprentice['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                                  value      = 1,
                                                  inplace    = True)


# AVG_PREP_VID_TIME
apprentice['OUT_AVG_PREP_VID_TIME'] = 0
condition_hi  = apprentice.loc[0:,'OUT_AVG_PREP_VID_TIME'][apprentice['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]
condition_low = apprentice.loc[0:,'OUT_AVG_PREP_VID_TIME'][apprentice['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_LOW]

apprentice['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

apprentice['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_low,
                                            value      = 1,
                                            inplace    = True)


# AVG_CLICKS_PER_VISIT
apprentice['OUT_AVG_CLICKS_PER_VISIT'] = 0
condition_hi  = apprentice.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][apprentice['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]
condition_low = apprentice.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][apprentice['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_LOW]

apprentice['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

apprentice['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_low,
                                               value      = 1,
                                               inplace    = True)


# TOTAL_MEALS_ORDERED
apprentice['OUT_TOTAL_MEALS_ORDERED'] = 0
condition_hi = apprentice.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][apprentice['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

apprentice['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)


# UNIQUE_MEALS_PURCH
apprentice['OUT_UNIQUE_MEALS_PURCH'] = 0
condition_hi = apprentice.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][apprentice['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HI]

apprentice['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)


# CONTACTS_W_CUSTOMER_SERVICE
apprentice['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi  = apprentice.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][apprentice['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]
condition_low = apprentice.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][apprentice['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_LOW]

apprentice['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                                      value      = 1,
                                                      inplace    = True)

apprentice['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_low,
                                                      value      = 1,
                                                      inplace    = True)


# CANCELLATIONS_BEFORE_NOON
apprentice['OUT_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = apprentice.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][apprentice['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_HI]

apprentice['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                                    value      = 1,
                                                    inplace    = True)


# CANCELLATIONS_AFTER_NOON
apprentice['OUT_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = apprentice.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][apprentice['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

apprentice['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)


# PC_LOGINS
apprentice['OUT_PC_LOGINS'] = 0
condition_hi  = apprentice.loc[0:,'OUT_PC_LOGINS'][apprentice['PC_LOGINS'] > PC_LOGINS_HI]
condition_low = apprentice.loc[0:,'OUT_PC_LOGINS'][apprentice['PC_LOGINS'] < PC_LOGINS_LOW]

apprentice['OUT_PC_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

apprentice['OUT_PC_LOGINS'].replace(to_replace = condition_low,
                                    value      = 1,
                                    inplace    = True)


# MOBILE_LOGINS
apprentice['OUT_MOBILE_LOGINS'] = 0
condition_hi  = apprentice.loc[0:,'OUT_MOBILE_LOGINS'][apprentice['MOBILE_LOGINS'] > MOBILE_LOGINS_HI]
condition_low = apprentice.loc[0:,'OUT_MOBILE_LOGINS'][apprentice['MOBILE_LOGINS'] < MOBILE_LOGINS_LOW]

apprentice['OUT_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)

apprentice['OUT_MOBILE_LOGINS'].replace(to_replace = condition_low,
                                        value      = 1,
                                        inplace    = True)


# WEEKLY_PLAN
apprentice['OUT_WEEKLY_PLAN'] = 0
condition_hi = apprentice.loc[0:,'OUT_WEEKLY_PLAN'][apprentice['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]

apprentice['OUT_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)


# EARLY_DELIVERIES
apprentice['OUT_EARLY_DELIVERIES'] = 0
condition_hi = apprentice.loc[0:,'OUT_EARLY_DELIVERIES'][apprentice['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]

apprentice['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)


# LATE_DELIVERIES
apprentice['OUT_LATE_DELIVERIES'] = 0
condition_hi = apprentice.loc[0:,'OUT_LATE_DELIVERIES'][apprentice['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

apprentice['OUT_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                          value      = 1,
                                          inplace    = True)


# FOLLOWED_RECOMMENDATIONS_PCT
apprentice['OUT_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition_hi = apprentice.loc[0:,'OUT_FOLLOWED_RECOMMENDATIONS_PCT'][apprentice['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_HI]

apprentice['OUT_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                                       value      = 1,
                                                       inplace    = True)


# LARGEST_ORDER_SIZE
apprentice['OUT_LARGEST_ORDER_SIZE'] = 0
condition_hi  = apprentice.loc[0:,'OUT_LARGEST_ORDER_SIZE'][apprentice['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]
condition_low = apprentice.loc[0:,'OUT_LARGEST_ORDER_SIZE'][apprentice['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_LOW]

apprentice['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)

apprentice['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_low,
                                             value      = 1,
                                             inplace    = True)


# MASTER_CLASSES_ATTENDED
apprentice['OUT_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = apprentice.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][apprentice['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

apprentice['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                                  value      = 1,
                                                  inplace    = True)


# TOTAL_PHOTOS_VIEWED
apprentice['OUT_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = apprentice.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][apprentice['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_HI]

apprentice['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)






# Creating trend-based thresholds

AVG_TIME_PER_SITE_VISIT_CHANGE_HI         =     300          # data scatters above this point
AVG_PREP_VID_TIME_CHANGE_HI               =     300          # data scatters above this point
TOTAL_MEALS_ORDERED_CHANGE_HI             =     200          # data scatters above this point
TOTAL_PHOTOS_VIEWED_CHANGE_HI             =     500          # data scatters above this point
FOLLOWED_RECOMMENDATIONS_PCT_CHANGE_HI    =     30           # trend changes above this point
AVG_CLICKS_PER_VISIT_CHANGE_HI            =     11           # trend changes above this point
CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI     =     10           # trend changes above this point
LARGEST_ORDER_SIZE_CHANGE_HI              =     7            # trend changes above this point

WEEKLY_PLAN_CHANGE_AT                     =     0            # zero inflated
UNIQUE_MEALS_PURCH_CHANGE_AT              =     0            # zero inflated
PC_LOGINS_CHANGE_AT                       =     7            # different at 7
MOBILE_LOGINS_CHANGE_AT                   =     3            # different at 3
MASTER_CLASSES_ATTENDED_CHANGE_AT         =     3            # different at 3
TOTAL_PHOTOS_VIEWED_CHANGE_AT             =     0            # zero inflated


##############################################################################
## Feature Engineering (trend changes)                                      ##
##############################################################################

# developing features (columns) for outliers

########################################
## change above threshold             ##
########################################

# AVG_TIME_PER_SITE_VISIT
apprentice['CHANGE_AVG_TIME_PER_SITE_VISIT'] = 0
condition = apprentice.loc[0:,'CHANGE_AVG_TIME_PER_SITE_VISIT'][apprentice['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_CHANGE_HI]

apprentice['CHANGE_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                                     value      = 1,
                                                     inplace    = True)


# AVG_PREP_VID_TIME
apprentice['CHANGE_AVG_PREP_VID_TIME'] = 0
condition = apprentice.loc[0:,'CHANGE_AVG_PREP_VID_TIME'][apprentice['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_CHANGE_HI]

apprentice['CHANGE_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                               value      = 1,
                                               inplace    = True)


# TOTAL_MEALS_ORDERED
apprentice['CHANGE_TOTAL_MEALS_ORDERED'] = 0
condition = apprentice.loc[0:,'CHANGE_TOTAL_MEALS_ORDERED'][apprentice['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_CHANGE_HI]

apprentice['CHANGE_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                                 value      = 1,
                                                 inplace    = True)


# TOTAL_PHOTOS_VIEWED
apprentice['CHANGE_TOTAL_PHOTOS_VIEWED'] = 0
condition = apprentice.loc[0:,'CHANGE_TOTAL_PHOTOS_VIEWED'][apprentice['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_CHANGE_HI]

apprentice['CHANGE_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                                 value      = 1,
                                                 inplace    = True)


# FOLLOWED_RECOMMENDATIONS_PCT
apprentice['CHANGE_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition = apprentice.loc[0:,'CHANGE_FOLLOWED_RECOMMENDATIONS_PCT'][apprentice['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_CHANGE_HI]

apprentice['CHANGE_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition,
                                                          value      = 1,
                                                          inplace    = True)


# AVG_CLICKS_PER_VISIT
apprentice['CHANGE_AVG_CLICKS_PER_VISIT'] = 0
condition = apprentice.loc[0:,'CHANGE_AVG_CLICKS_PER_VISIT'][apprentice['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_CHANGE_HI]

apprentice['CHANGE_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                                  value      = 1,
                                                  inplace    = True)


# CONTACTS_W_CUSTOMER_SERVICE
apprentice['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = apprentice.loc[0:,'CHANGE_CONTACTS_W_CUSTOMER_SERVICE'][apprentice['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI]

apprentice['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                                         value      = 1,
                                                         inplace    = True)


# LARGEST_ORDER_SIZE
apprentice['CHANGE_LARGEST_ORDER_SIZE'] = 0
condition = apprentice.loc[0:,'CHANGE_LARGEST_ORDER_SIZE'][apprentice['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_CHANGE_HI]

apprentice['CHANGE_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)




########################################
## change at threshold                ##
########################################

# WEEKLY_PLAN
apprentice['CHANGE_WEEKLY_PLAN'] = 0
condition = apprentice.loc[0:,'CHANGE_WEEKLY_PLAN'][apprentice['WEEKLY_PLAN'] == WEEKLY_PLAN_CHANGE_AT]

apprentice['CHANGE_WEEKLY_PLAN'].replace(to_replace = condition,
                                         value      = 1,
                                         inplace    = True)


# TOTAL_PHOTOS_VIEWED
apprentice['CHANGE2_TOTAL_PHOTOS_VIEWED'] = 0
condition = apprentice.loc[0:,'CHANGE2_TOTAL_PHOTOS_VIEWED'][apprentice['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_CHANGE_AT]

apprentice['CHANGE2_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                                  value      = 1,
                                                  inplace    = True)


# UNIQUE_MEALS_PURCH
apprentice['CHANGE_UNIQUE_MEALS_PURCH'] = 0
condition = apprentice.loc[0:,'CHANGE_UNIQUE_MEALS_PURCH'][apprentice['UNIQUE_MEALS_PURCH'] == UNIQUE_MEALS_PURCH_CHANGE_AT]

apprentice['CHANGE_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)


# PC_LOGINS
apprentice['CHANGE_PC_LOGINS'] = 0
condition = apprentice.loc[0:,'CHANGE_PC_LOGINS'][apprentice['PC_LOGINS'] == PC_LOGINS_CHANGE_AT]

apprentice['CHANGE_PC_LOGINS'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# MOBILE_LOGINS
apprentice['CHANGE_MOBILE_LOGINS'] = 0
condition = apprentice.loc[0:,'CHANGE_MOBILE_LOGINS'][apprentice['MOBILE_LOGINS'] == MOBILE_LOGINS_CHANGE_AT]

apprentice['CHANGE_MOBILE_LOGINS'].replace(to_replace = condition,
                                         value      = 1,
                                         inplace    = True)


# MASTER_CLASSES_ATTENDED
apprentice['CHANGE_MASTER_CLASSES_ATTENDED'] = 0
condition = apprentice.loc[0:,'CHANGE_MASTER_CLASSES_ATTENDED'][apprentice['MASTER_CLASSES_ATTENDED'] == MASTER_CLASSES_ATTENDED_CHANGE_AT]

apprentice['CHANGE_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition,
                                                     value      = 1,
                                                     inplace    = True)







# one hot encoding categorical variables
ONE_HOT_DOMAIN_GROUP = pd.get_dummies(apprentice['DOMAIN_GROUP'])
ONE_HOT_TASTES_AND_PREFERENCES = pd.get_dummies(apprentice['TASTES_AND_PREFERENCES'])



# dropping categorical variables after they've been encoded
apprentice = apprentice.drop('DOMAIN_GROUP', axis = 1)
apprentice = apprentice.drop('TASTES_AND_PREFERENCES', axis = 1)


# joining codings together
apprentice = apprentice.join([ONE_HOT_DOMAIN_GROUP, ONE_HOT_TASTES_AND_PREFERENCES])


# saving new columns
new_columns = apprentice.columns



apprentice.columns = ['REVENUE', 'CROSS_SELL_SUCCESS', 'NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME',
                      'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
                      'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER',           
                      'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON', 'PC_LOGINS',
                      'MOBILE_LOGINS','WEEKLY_PLAN','EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER',
                      'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME',                  
                      'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING',                
                      'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED', 'EMAIL_DOMAIN', 'OUT_AVG_TIME_PER_SITE_VISIT',
                      'OUT_AVG_PREP_VID_TIME', 'OUT_AVG_CLICKS_PER_VISIT', 'OUT_TOTAL_MEALS_ORDERED',
                      'OUT_UNIQUE_MEALS_PURCH', 'OUT_CONTACTS_W_CUSTOMER_SERVICE', 'OUT_CANCELLATIONS_BEFORE_NOON',
                      'OUT_CANCELLATIONS_AFTER_NOON', 'OUT_PC_LOGINS', 'OUT_MOBILE_LOGINS', 'OUT_WEEKLY_PLAN',
                      'OUT_EARLY_DELIVERIES','OUT_LATE_DELIVERIES', 'OUT_FOLLOWED_RECOMMENDATIONS_PCT',
                      'OUT_LARGEST_ORDER_SIZE', 'OUT_MASTER_CLASSES_ATTENDED', 'OUT_TOTAL_PHOTOS_VIEWED',
                      'CHANGE_AVG_TIME_PER_SITE_VISIT', 'CHANGE_AVG_PREP_VID_TIME', 'CHANGE_TOTAL_MEALS_ORDERED',
                      'CHANGE_TOTAL_PHOTOS_VIEWED', 'CHANGE_FOLLOWED_RECOMMENDATIONS_PCT',         
                      'CHANGE_AVG_CLICKS_PER_VISIT', 'CHANGE_CONTACTS_W_CUSTOMER_SERVICE',
                      'CHANGE_LARGEST_ORDER_SIZE', 'CHANGE_WEEKLY_PLAN', 'CHANGE2_TOTAL_PHOTOS_VIEWED',
                      'CHANGE_UNIQUE_MEALS_PURCH', 'CHANGE_PC_LOGINS','CHANGE_MOBILE_LOGINS',      
                      'CHANGE_MASTER_CLASSES_ATTENDED', 'JOB', 'JUNK', 'PERSONAL', 
                      'TASTES_AND_PREFERENCES_NO', 'TASTES_AND_PREFERENCES_YES']  # re-labelling Email Domains, and
                                                                                  # TASTES_AND_PREFERENCES
                                                                                  # easier to understand


candidate_dict = {

 # model 1 (p-values < 0.05)
 'model1'     : ['TOTAL_MEALS_ORDERED', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON',
                 'MOBILE_LOGINS', 'FOLLOWED_RECOMMENDATIONS_PCT', 'CHANGE_TOTAL_MEALS_ORDERED',
                 'JOB', 'JUNK', 'TASTES_AND_PREFERENCES_YES']}



################################################################################
# Train/Test Split
################################################################################

# train/test split with the full model
apprentice_data   =  apprentice.loc[ : , candidate_dict['model1']]
apprentice_target =  apprentice.loc[ : , 'CROSS_SELL_SUCCESS']


# This is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            apprentice_data,
            apprentice_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = apprentice_target)





################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# declaring a hyperparameter space
#criterion_space = ['gini', 'entropy']
#splitter_space = ['best', 'random']
#depth_space = pd.np.arange(1, 25)
#leaf_space  = pd.np.arange(1, 100)


# creating a hyperparameter grid
#param_grid = {'criterion'        : criterion_space,
#              'splitter'         : splitter_space,
#              'max_depth'        : depth_space,
#              'min_samples_leaf' : leaf_space}


# INSTANTIATING the model object without hyperparameters
#tuned_tree = DecisionTreeClassifier(random_state = 222)


# GridSearchCV object
#tuned_tree_cv = GridSearchCV(estimator  = tuned_tree,
#                             param_grid = param_grid,
#                             cv         = 3,
#                             scoring    = make_scorer(roc_auc_score,
#                                                      needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
#tuned_tree_cv.fit(apprentice_data, apprentice_target)


# PREDICT step is not needed


# printing the optimal parameters and best score
#print("Tuned Parameters  :", tuned_tree_cv.best_params_)
#print("Tuned Training AUC:", tuned_tree_cv.best_score_.round(4))




# building a model based on hyperparameter tuning results, extracted from a GridSearchCV fill 
# with a Decision Tree Classifier;


# INSTANTIATING a decision tree model with tuned values
tree_tuned = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=16, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=222, splitter='random')


# FIT step is not needed
tree_tuned_fit  = tree_tuned.fit(apprentice_data, apprentice_target)

# PREDICTING based on the testing set
tree_tuned_pred = tree_tuned_fit.predict(X_test)



################################################################################
# Final Model Score (score)
################################################################################

# SCORING the results
print('Training ACCURACY:', tree_tuned.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', tree_tuned.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = tree_tuned_pred).round(4))


# In[22]:


#print(datetime.now() - startTime)


# In[ ]:





# In[ ]:




