# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from six import StringIO  
import pydotplus
from sklearn import tree

def display_decision_tree(clf, feature_names):
    """Affiche l'arbre de décision."""
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    display(Image(graph.create_png(), width=20))

def display_styled_dataframe(df_predictions):
    """Affiche un DataFrame stylisé avec une coloration conditionnelle."""
    def color_probabilities(val):
        try:
            float_val = float(val)
            color = plt.cm.RdYlGn(float_val)
            return f'background-color: rgba({",".join(map(str, (int(c*255) for c in color[:3])))},.5)'
        except ValueError:
            return ''

    styled_df = df_predictions.style.applymap(color_probabilities, subset=[col for col in df_predictions.columns if col.startswith('Prob')])
    
    styled_df = styled_df.set_properties(**{
        'border-color': 'black',
        'border-style': 'solid',
        'border-width': '1px',
        'text-align': 'center',
        'padding': '10px',
        'font-size': '11pt'
    })

    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#4CAF50'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding-left', '10px'),
        ]},
        {'selector': 'td', 'props': [
            ('padding-left', '25px'),
        ]},
    ])
    
    display(styled_df)
