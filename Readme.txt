## Recommendation System based on HCI and AI

### Project Description
The artificial intelligence techniques and the human computer interaction concepts were applied in the development of recommendation system. Google Play Store dataset was used for generating application suggestions. The use of content based filtering using TF IDF and cosine similarity generated similar application suggestions.

### Features
Recommending similar applications
Recommending applications by category
Showing popular applications
Interactive based menu interface.
Human computer interaction (text based).

### Dataset
A dataset with Google Play Store was obtained in Kaggle. The data includes the name of applications, category, genre, rating, and review.

### Technologies Used
Python
Pandas
NumPy
Scikit learn
TF IDF Vectorization
Cosine Similarity

### System Workflow
Loading dataset
Removing missing and duplicate data.
Transforming the rating values to numerical ones.
A mixture of category and genre characteristics.
TF IDF matrix construction.
Calculating cosine similarity
Generating recommendations
Presentation of findings in interactive menu.

### How to Run
Download project files
Put the file known as googleplaystore.csv in the project folder.
Open Python environment
Run recommendation_system.py file
Select menu options
Give the name or the category of the application.

### Sample Execution
#### Input
Select Option: 1
Name of the application: Instagram.

#### Output
Recommended Apps:
Facebook Lite
Messenger
Tumblr
Snapchat
Pinterest

### Human Computer Interaction
Interactive text based interface was applied. Navigation was menu driven with the choice of type of recommendation. Recommendations were created dynamically depending on user input.

### Learning Outcomes
Understanding recommendation systems
Implementation of content based filtering.
TF IDF and Cosine similarity.
Adopting interactive smart system.
Performing data preprocessing

### Future Improvements
Graphical user interface to be added.
Adding collaborative filtering
Deploying web application
Including the user preference learning.
Introduction of hybrid recommendation system.