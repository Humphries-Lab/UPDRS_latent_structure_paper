# figure_config.py

import matplotlib.pyplot as plt
from cmocean import cm

# Define figure properties for the paper
fig_properties = {
    'fontsize': 12,
    'largefontsize': 14,
    'labelsize':12,
    'barlabelsize':6,
    'fontname': 'Arial',
    'marker_size': 3,
    'big_marker_size':10,
    'marker_style': 'o',
    'dpi': 600,
    'color_mode': 'rgb',
    'units': 'centimeters',
    'line_widths': {
        'plot': 1,
        'error': 0.5,
        'axis': 0.5
    },
    'figsize': {
        'square': (4, 4),
        'linkedscatter': (3, 5),
        'dists': (8, 4),
        'violin': (6, 4),
        'scatter3D': (6, 6),
        'loadings':(10,6),
        'wider':(14,6)
    },
    'color_maps': {
        'rate_map': cm.gray,  # Greys equivalent
        'inhibit_map': cm.deep,  # Blues equivalent
        'excite_map': cm.thermal,  # OrRd equivalent
        'PC_map': cm.balance,  # PuOr equivalent
    },
    'colors': {
        'highlight': [0.8, 0.4, 0.2],
        'baseline': [0.7, 0.7, 0.7],
        'inhibition': [0.4, 0.4, 0.8],
        'text': [0.3, 0.3, 0.3],
        'red': [1, 0, 0],
        'blue': [0, 0, 1],
        'green': [0, 1, 0],
        'yellow': [1, 1, 0],
        'black': [0, 0, 0]
        
    },
    'export_path': './New_Panels/'
}

# Function to apply figure properties
def apply_figure_properties(fig, properties):
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(properties['fontsize'])
        ax.yaxis.label.set_fontsize(properties['fontsize'])
        ax.xaxis.label.set_fontname(properties['fontname'])
        ax.yaxis.label.set_fontname(properties['fontname'])
        ax.tick_params(axis='both', which='major', labelsize=properties['fontsize'], width=properties['line_widths']['axis'])

        # Set line widths for the axis box
        ax.spines['top'].set_linewidth(properties['line_widths']['axis'])
        ax.spines['right'].set_linewidth(properties['line_widths']['axis'])
        ax.spines['bottom'].set_linewidth(properties['line_widths']['axis'])
        ax.spines['left'].set_linewidth(properties['line_widths']['axis'])

    return fig
