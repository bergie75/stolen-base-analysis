The code included was designed to predict the outcome of a stolen base attempt (a runner on one base attempting to move to the next without being tagged out by the defense), which is encoded as a binary variable. An ensemble machine learning model (using gradient-boosted trees) and different bootstrap samples of the data were used, because the success rate for stolen bases leads to an unbalanced dataset.

Roughly speaking, the code for training models is contained in the file mariners_ensemble.py,
while the code for visualizations and use of the model for player improvement is contained in mariners_visualizations.py. My solution for the second problem
makes use of my model, and the necessary methods are imported where needed. A rough summary of the methods in
these files can be found below.

########################################################################################################################

MARINERS_ENSEMBLE.PY

create_predictions: a method to put everything together and generate the predictions. This takes in a dataframe
representing the test data, a list of models in the ensemble, and the weights.

ensemble_predict: a method to take in a list of models and weights and return a prediction. When use_prob=True,
the third variable returned is the ensemble's prediction for the probability of being caught stealing.

display_confusion_matrix: a method to generate confusion matrices for a model.

clean_split_data: the function that handles pre-processing of the data, called during bootstrap_ensemble

bootstrap_ensemble: makes use of other functions to prepare data and train the model. Returns a list of models, weights,
and a Boolean indicating whether scaling was used for convenience. This Boolean is not saved.

save_model, load_model, delete_model: user functions to handle saving and loading models or deleting
outdated models

return_variables: a helper method to give mariners_visualizations.py access to the variables to remove from a dataset

########################################################################################################################

MARINERS_VISUALIZATIONS.PY

explore_relationships: a method to examine scatterplots of variables vs is_cs

scatter_ind_group: a method to show scatterplots of two variables for an individual player and the rest of the data

ind_vs_group_hist: creates histograms of a particular stat for a player and the larger group

scroll_charts: a helper method to call ind_vs_group_hist repeatedly on all stats

check_improvements: the method used to perform the comparisons discussed in the coaching recommendations,
replacing one stat with randomly sampled values and applying the model from problem 1 to analyze the different
results
