import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    fig.tight_layout()
    # fig.savefig('./images/'+title+'.png')
    return

# Function for plotting the test metrics with respect to cutoff values
def plot_metrics(y_labels,probabilities,cuts):
    '''This function is used to plot the test metrics with respect to an array of cutoff values.

    Arguments:
    ---------
    y_labels: [np.array,list], array containing labels for each patch image in the dataset
    probabilities: [np.array,list], array containing the model predictions for each patch image in the dataset
    cuts: [np.array,list], array containing the cutoff values intended to plot 
    
    Output:
    fig: plt.Figure, figure of the test metrics with respect to the array of cutoff values

    '''

    # Initialize arrays for accuracy, precision, recall, and f1-score
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for cut in cuts:
        
        # Classifications based on the cut values given
        y_predicts = [1 if prob > cut else 0 for prob in probabilities]

        # Create the confusion matrix with the predictions and true labels
        cf_matrix = confusion_matrix(y_labels,y_predicts)


        # Metrics for Binary Confusion Matrices
        accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
        precision = cf_matrix[1,1] / sum(cf_matrix[:,1])
        recall = cf_matrix[1,1] / sum(cf_matrix[1,:])
        f1_score = 2*precision*recall / (precision + recall)

        # Adding the metrics to lists of metrics with different cut values
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # Plotting the test metrics with respect to the cutoff values
    fig,ax = plt.subplots(figsize = (8,8))
    ax.plot(cuts,accuracies,label='accuracy')
    ax.plot(cuts,precisions,label='precision')
    ax.plot(cuts,recalls,label='recall')
    ax.plot(cuts,f1_scores,label='F1-score')
    ax.set_title('Optimizing Cuttoff Values')
    ax.set_xlabel('Cutoff Value')
    ax.set_ylabel('Evaluation Metrics in Decimal')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('./images/denseNet201_optimization.png')
    return fig

def plot_rocs(models,test_data_path):
    
    # Initialize the figure and plot the 50% line
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot([0, 1], [0, 1], 'k--')

    # For loop to loop through the models, make predictions for each model and plot the ROC curve
    for k,v in models.items():

        # Initializing the test generator using the target size for the model 
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_data_path, 
        target_size = v[1], 
        batch_size = 32,shuffle=False,class_mode='binary')
        y_test = test_generator.labels


        # Making the predictions on the test set
        probs = v[0].predict(test_generator)
        probs_list = [prob[0] for prob in probs]
        # Calculating the false positive rate and true positive rate
        fpr, tpr, thresholds = roc_curve(y_test, probs_list)

        # if tpr/fpr < 0.5:
        #     ax.plot(1 - fpr, 1 - tpr, label=k + '(area = {:.3f})'.format(auc(1 - fpr,1 - tpr)))
        # else:
        ax.plot(fpr, tpr, label= k + '(area = {:.3f})'.format(auc(fpr, tpr)))

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    ax.legend(loc='best')
    fig.tight_layout()
    # fig.savefig('./images/ROC_comparisons.png')
    return fig