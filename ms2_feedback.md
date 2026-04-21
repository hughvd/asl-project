# Directions

Objective:

In this milestone, your focus should be on a well-motivated Exploratory Data Analysis (EDA) and initial modeling. The objective is to interpret findings from your EDA and use them to motivate your initial model choices and possibly refine and solidify the project goals. This stage will also involve creating a clear navigation plan for the project's final deliverables. Groups will also establish an initial, straightforward model or set of models, referred to as the baseline models. These models serves as a benchmark for measuring progress in subsequent iterations. Additionally, students will propose a comprehensive pipeline for model training and testing. The pipeline encompasses data preprocessing, model implementation, and the evaluation of model outputs using a well-considered set of metrics. This preparation lays the foundation for more advanced model iterations.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Deliverables:

An 8-10 minute presentation with your TF, with roughly 10 minutes of follow-up questions and conversation with your TF.
The slides used in your presentation (pdf).
Your .ipynb notebook that was used to access the data and perform any data wrangling, EDA, and visualizations.
Grading emphasizes quality and thoroughness of thinking over visual polish, but slides should still be clear and readable.  See Slide Tips below.  Grading will also take into account your responses to TF questioning.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Details:

Presentation (and Slides):

Required to be 8-10 minutes in length.  Expect roughly 10 minutes of questions of follow-up.  Earmark 25 minutes in total.
Should be on Zoom with your time slot arranged with TF (can be in-person if everyone involved agrees, including your TF).  
Required to use a slide deck (can be created using any software) that will be submitted as a pdf.
All team members must be present with roughly equal contribution/time presenting.
Can be more conversational and less polished/scripted. 
Suggested Structure (not all sections will have equal time):
Introduction and motivation of the data and project.
Data: details about acquisition/source, wrangling/preparation, visualizations/EDAs, and any insights gained from these steps.
Redefinition and rescoping of the problem statement.
Delineation of next steps by team members.
Future considerations: including potential class of model(s) considered, potential challenges and concerns, and possible open questions for your TF.
Notebook:

Please submit the .ipynb notebook that you used to access the data and perform any data wrangling, EDA, and visualizations that you performed, which will include the items outlined below:

Canvas Project number
Group members' names
Data Description
Summary of the Data + Data Analysis + Meaningful Insights
Clean and Labeled Visualizations
Summary of Findings based on Data 
Clear Research Question
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Milestone Checklist: Your submission could include any of the following components. Note that these guidelines should serve as a suggestion for what to consider including and do not apply to every project. If you have any concerns, please contact your TF in advance. 

Access: Download, collect, or scrape* the dataset from relevant source(s). In other words, can you get the data on your local machine or in a cloud based repository?
Load: Start a new Jupyter Notebook, import necessary Python libraries (e.g., pandas, numpy, sklearn, umap, torch, etc.) and load your dataset. 
Understand: Examine the dataset. Ensure you understand what different columns/rows represent or the image/text intricacies. Be sure to report the total size of your data in megabytes or gigabytes.  
Preprocess: Propose or perform basic dataset cleaning to make it suitable for analysis, visualization, and modeling which you will pursue in later milestones. Document each step in your Jupyter Notebook to justify the preprocessing decisions made. Reference the next sections for details on what comprehensive data cleaning and preprocessing should include. 
Summarize: Provide the shape of the data, data types, and descriptive statistics such as mean, max, and dtypes. Additionally, provide a summary of the features of the data, including histograms, correlation plots, and clustering plots as appropriate. 
Analyze : Identify patterns, trends, and outliers in the data. Additionally, explore the relationships between variables and identify any potential confounding or unmeasured variables/features that may impact the analysis.
Visualize: Visualizations are important components of EDA and should be clean, labeled, and well-presented. You need to ensure that your visualizations are easy to understand and can be included in their final presentation slides or report. Anyone that reads your EDA should be able to understand what is depicted in the plots just by looking at them. Visualizations should be readable well-labeled.
Gain Insights**: Based on your analysis of the data, provide meaningful insights.  A meaningful insight is one that connects directly to your project question and could influence your modeling decisions.  Any insights should be well-supported by the data, provide actionable recommendations, and have a brief justification for why or how it’s important to the project.  Think: are there any other data sources that you will need to access in the future to supplement what you have already gathered?
Rescope and Revise: Based on the insights gained through EDA, you should develop a clear project/research question that will guide your analysis. This question should be well-defined, specific to the problem at hand, and an improvement from the initial project proposal.
*You must have a usable dataset already collected. Projects based only on planned data collection will not receive credit for this milestone.

**For a simple example: if you notice that your images are of different sizes, you may need to do adjust the resolution so all images are of standard size.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

Cleaning and Preprocessing:

Below is a non-exhaustive list of issues you might want to check for and address in your dataset preprocessing and wrangling.

Missing Data: 

Missing data may arise due to a range of factors, such as human error (e.g., intentional non-response to survey questions), malfunctioning electrical sensors, or other causes. When data is missing, a significant amount of valuable information can be lost. Investigate the extent and pattern of missing data. Determine the nature of missingness (Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR)) , these are CS1090a concepts, and apply the most suitable technique to address it. Options include data deletion, mean/mode imputation, or more advanced methods like multiple imputation or k-NN imputation. Justify your choice based on the dataset's characteristics.

Anomaly Detection: 

Anomalies are data points that deviate significantly from the rest of the dataset. These may arise from data entry errors, sensor malfunctions, or rare but valid events. If not addressed, anomalies can distort statistical summaries and negatively impact model performance. Identify potential anomalies using appropriate methods.. Once identified, determine whether they are errors/noise, in which case they may be removed or corrected. If valid, consider keeping them, especially if they are meaningful for the task. Clearly justify your decision to remove, transform, or retain anomalies based on their impact on the dataset and project goals.

Data Imbalance:

Imbalanced data is a common issue in classification problems when one class has significantly fewer samples than the other. When dealing with imbalanced data, machine learning models may learn to favor the majority class and make predictions that prioritize accuracy for that class. This can result in unsatisfactory performance for the minority class and reduced overall model effectiveness.

Assess the class distribution in your dataset, especially for classification tasks. If a significant imbalance is present, apply resampling techniques (oversampling minorities or under sampling majorities) or apply synthetic data generation methods like SMOTE to achieve a balanced dataset, another CS1090a content piece.

Feature Scaling:

Scaling the data is a crucial step in improving model performance and avoiding bias, as well as enhancing interpretability. When features are not appropriately scaled, those with larger scales can potentially dominate the analysis and result in biased conclusions. Standardize or normalize numerical features to ensure equal weighting in analytical models. Choose and apply the most appropriate scaling method (e.g., Min-Max normalization, Z-score standardization) based on your data distribution and the models you plan to use.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

Slide Tips 

Make plots readable — check font sizes carefully
Label everything clearly — axes, titles, legends
Use fewer words — slides are not paragraphs
One idea per slide — avoid “NPC slides” that add noise
Add slide numbers (Pavlos really wants this)

# Feedback

1. Problem definition (0.75/1.0): The research question was clearly defined, but it is missing from the notebook. The notebook should be structured more like a report with clearly labeled sections and key components included.
2. Motivation (1.0/1.0)
3. Data access & provenance (0.5/1.0): The dataset is described clearly, but the source and access details are not fully specified. The notebook / presentation should include the data source, a link, acquisition details.
4. Data loading (1.0/1.0)
5. Dataset understanding (1.0/1.0)
6. Summary / initial EDA (1.0/1.0)
7. Preprocessing / data preparation (0.5/1.0): Preprocessing steps are good, but the justification is only partially discussed. This should be more explicitly explained and documented in the notebook.
8. Analysis / patterns (0.3/1.0): Identifies the data leakage issue, but overall analysis of patterns, relationships, and potential confounders is limited. More thorough exploration and analysis of the data is needed.
9. Visualization quality (0.75/1.0): Visualizations are clear and readable, but they do not clearly highlight specific insights. Adding brief interpretation or takeaways would strengthen the EDA.
10. Insights & next steps (0.75/1.0): Next steps are well motivated, especially the plan to collect additional data to address dataset limitations. However, the connection between insights from the data and these decisions could be made more explicit.
11. Rescope / adaptation (1.0/1.0)
12. Presentation & slides (0.75/1.0): Slides are clear and generally well organized, but some are text-heavy and do not always highlight key takeaways.