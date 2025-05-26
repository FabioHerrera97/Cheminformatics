import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, precision_score, f1_score

class PlotEvaluation:
    '''
    A class for visualizing and comparing model performance metrics.
    
    This class generates a multi-panel figure comparing classification model performance
    across accuracy, recall, precision, F1-score, and ROC curves. The visualization is 
    designed to facilitate quick comparison between multiple models.

    Attributes:
        y_true (array-like): Ground truth (correct) target values.
        y_preds (list of array-like): List of predicted target values for each model.
        y_probas (list of array-like): List of predicted probabilities for each model.
        model_names (list of str): List of model names corresponding to predictions.
    '''
    
    def __init__(self, y_true, y_preds, y_probas, model_names):
        self.y_true = y_true
        self.y_preds = y_preds
        self.y_probas = y_probas
        self.model_names = model_names

    def plot_metrics(self):
        '''
        Generate and display a multi-panel comparison plot of model metrics.
        
        Creates a 2x3 grid containing:
        - Top-left: Accuracy comparison (bar plot)
        - Top-center: Precision comparison (bar plot)
        - Top-right: Recall comparison (bar plot)
        - Bottom-left: F1-score comparison (bar plot)
        - Bottom-center: ROC curves with AUC scores
        - Bottom-right: Empty (turned off)
        
        The plot includes:
        - All metrics scaled from 0 to 1
        - ROC curves with AUC values in the legend
        - Automatic layout adjustment for readability
        '''
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Performance Comparison', fontsize=18, y=1.02)
        
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        
        # 1. Accuracy Plot
        for i, (name, y_pred) in enumerate(zip(self.model_names, self.y_preds)):
            acc = accuracy_score(self.y_true, y_pred)
            axes[0, 0].bar(name, acc, color=colors[0])
        axes[0, 0].set_title('Accuracy Score', fontsize=14)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Precision Plot
        for i, (name, y_pred) in enumerate(zip(self.model_names, self.y_preds)):
            prec = precision_score(self.y_true, y_pred)
            axes[0, 1].bar(name, prec, color=colors[1])
        axes[0, 1].set_title('Precision Score', fontsize=14)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Recall Plot
        for i, (name, y_pred) in enumerate(zip(self.model_names, self.y_preds)):
            rec = recall_score(self.y_true, y_pred)
            axes[0, 2].bar(name, rec, color=colors[2])
        axes[0, 2].set_title('Recall Score', fontsize=14)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4. F1-Score Plot
        for i, (name, y_pred) in enumerate(zip(self.model_names, self.y_preds)):
            f1 = f1_score(self.y_true, y_pred)
            axes[1, 0].bar(name, f1, color=colors[3])
        axes[1, 0].set_title('F1 Score', fontsize=14)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 5. ROC Curve Plot
        for i, (name, y_proba) in enumerate(zip(self.model_names, self.y_probas)):
            fpr, tpr, _ = roc_curve(self.y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            axes[1, 1].plot(fpr, tpr, lw=2, 
                          label=f'{name} (AUC = {roc_auc:.3f})')
        axes[1, 1].set_title('ROC Curves', fontsize=14)
        axes[1, 1].legend(loc='lower right', fontsize=10)
        axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1)
        
        axes[1, 2].axis('off')
        
        plt.tight_layout(pad=3.0)
        
        plt.figtext(0.5, 0.01, 
                   'Classification Metrics Comparison', 
                   ha='center', fontsize=12, style='italic')
        
        plt.show()