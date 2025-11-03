import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use('science')
except:
    pass
plt.rcParams.update({'font.size': 16,       # Default font size for general text
                    'axes.titlesize': 16,   # Font size for plot titles
                    'axes.labelsize': 16,   # Font size for axis labels
                    'xtick.labelsize': 14,  # Font size for x-axis tick labels
                    'ytick.labelsize': 14,  # Font size for y-axis tick labels
                    'legend.fontsize': 14,  # Font size for legend
                    'xtick.minor.visible': False, # Minor x ticks
                    'ytick.minor.visible': False, # Minor y ticks
                    'font.family': 'serif', # Set the font family to 'serif'
                    'font.serif': ['cmr10'], # Specify the Computer Modern Roman font (cmr10)
                    'mathtext.fontset': 'cm', # Ensure minus signs in mathematical text are rendered correctly
                    'axes.formatter.use_mathtext': True # Fixes issues with minus signs in tick labels
                    })

# plt.rcParams['font.serif'] = ['Computer Modern Serif'] + plt.rcParams['font.serif']
