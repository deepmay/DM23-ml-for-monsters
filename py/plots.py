import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import string

from networkx.drawing.nx_agraph import graphviz_layout

def make_color_map_without_replacement(G):
    color_map = []
    for node in G.nodes():
        if node == 'start':
            color_map.append('white')
        # 1st tertile
        elif node in ['A', 'A3', 'A22', 'A211', 'A12', 'A111',
                      'D', 'D3', 'D22', 'D211', 'D12', 'D111',
                      'C3', 'C2', 'C32', 'C311', 'C22', 'C211', 'C12', 'C121', 'C11', 'C111'
                      'B3', 'B2', 'B32', 'B311', 'B22', 'B211', 'B12', 'B121', 'B11', 'B111']:
            color_map.append('blue')
        else:
            color_map.append('white')
    return color_map

def make_size_map_without_replacement(G):
    size_map = []
    for node in G.nodes():
        if node == 'start':
            size_map.append(4000)
        if len(node) == 1:
            size_map.append(2500)
        if len(node) == 2:
            size_map.append(1000)
        if len(node) == 3:
            size_map.append(500)
        if len(node) == 4:
            size_map.append(250)
    return size_map

def make_lw_map_without_replacement(G):
    lw_map = []
    for node in G.nodes():
        if node == 'start':
            lw_map.append(0)
        else:
            lw_map.append(2)
    return lw_map

def plot_4_marbles_without_replacement():
    # Define the nodes and their hierarchy
    letters = string.ascii_uppercase[:4]
    nodes = [('start', letter) for letter in letters]
    indeces_1 = [1, 2, 3]
    indeces_2 = [1, 2]
    indeces_3 = [1]
    for parent, child in zip(letters, letters):
        for idx_1 in indeces_1:
            nodes.append((parent, f"{parent}{idx_1}"))
            for idx_2 in indeces_2:
                nodes.append((f"{parent}{idx_1}", f"{parent}{idx_1}{idx_2}"))
                for idx_3 in indeces_3:
                    nodes.append((f"{parent}{idx_1}{idx_2}", f"{parent}{idx_1}{idx_2}{idx_3}"))

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on the hierarchy
    for parent, child in nodes:
        G.add_edge(parent, child, weight=100)

    pos = graphviz_layout(G, prog="twopi", args='')
                          #args='-Gsplines=true -Gnodesep=0.6 -Goverlap=scalexy')

    color_map = make_color_map_without_replacement(G)
    size_map = make_size_map_without_replacement(G)
    lw_map = make_lw_map_without_replacement(G)


    # Plot the circular tree layout
    plt.figure(figsize=(12, 12))
    nodes = nx.draw_networkx_nodes(G, 
                                   pos=pos, 
                                   label=False,
                                   node_size=size_map,
                                   node_color=color_map,
                                   node_shape='o',
                                   linewidths=lw_map, 
                                   edgecolors='black')
    edges = nx.draw_networkx_edges(G, pos=pos)
    #labels = nx.draw_networkx_labels(G, 
    #                                 pos=pos)

    plt.axis("equal")
    plt.show()
    
def make_color_map_with_replacement(G):
    color1 = 'blue'
    color2 = 'white'
    color_map = []
    for node in G.nodes():
        if node == 'start':
            color_map.append(color2)
        # 1st tertile
        elif node.startswith('C') and (node.endswith('1') or
                                       node.endswith('2') or
                                       node.endswith('3')):
            color_map.append(color1)
        elif node in ['A','B','L'] or ((node.startswith('A') or
                                       node.startswith('B') or
                                       node.startswith('L')) and 
                                       (node.endswith('1') or
                                        node.endswith('2') or
                                        node.endswith('3'))):
            color_map.append(color1)
        # 2nd tertile
        elif (node.startswith('J') or node.startswith('K')) and (node.endswith('1') or node.endswith('2')):
            color_map.append(color1)
        elif node in ['H','I'] or ((node.startswith('H') or
                                   node.startswith('I')) and 
                                   (node.endswith('1') or
                                    node.endswith('2'))):
            color_map.append(color1)
        # 3rd tertile
        elif (node.startswith('G') or node.startswith('F') or node.startswith('E')) and node.endswith('1'):
            color_map.append(color1)
        elif node == 'D' or (node.startswith('D') and node.endswith('1')):
            color_map.append(color1)
        else:
            color_map.append(color2)
    return color_map
    
def make_size_map_with_replacement(G):
    size_map = []
    for node in G.nodes():
        if node == 'start':
            size_map.append(5000)
        if len(node) == 1:
            size_map.append(2000)
        if len(node) == 2:
            size_map.append(1000)
        if len(node) == 3:
            size_map.append(200)
    return size_map

def plot_4_marbles_with_replacement():

    # Define the nodes and their hierarchy
    letters = string.ascii_uppercase[:12]
    nodes = [('start', letter) for letter in letters]
    indeces_1 = [1, 2, 3, 4]
    indeces_2 = [1, 2, 3, 4]
    for parent, child in zip(letters, letters):
        for idx_1 in indeces_1:
            nodes.append((parent, f"{parent}{idx_1}"))
            for idx_2 in indeces_2:
                nodes.append((f"{parent}{idx_1}", f"{parent}{idx_1}{idx_2}"))

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on the hierarchy
    for parent, child in nodes:
        G.add_edge(parent, child, weight=100)

    pos = graphviz_layout(G, prog="twopi", args='')
                          #args='-Gsplines=true -Gnodesep=0.6 -Goverlap=scalexy')

    color_map = make_color_map_with_replacement(G)
    size_map = make_size_map_with_replacement(G)


    # Plot the circular tree layout
    plt.figure(figsize=(20, 20))
    nodes = nx.draw_networkx_nodes(G, 
                                   pos=pos, 
                                   label=True,
                                   node_size=size_map,
                                   node_color=color_map,
                                   node_shape='o',
                                   linewidths=2, 
                                   edgecolors='black')
    #labels = nx.draw_networkx_labels(G, 
    #                                 pos=pos)
    edges = nx.draw_networkx_edges(G, pos=pos)

    plt.axis("equal")

    #plt.axhline(pos['start'][1], color='black')
    #plt.axvline(pos['start'][0], color='black')
    # Calculate the range for each line segment
    # Set the angle and length of the lines
    angle_1 = 90  # Angle in degrees for the first line
    angle_2 = 210  # Angle in degrees for the second line
    angle_3 = 330  # Angle in degrees for the third line
    length = 20 + max(max(pos[node][0] - pos['start'][0]  for node in pos), 
                  max(pos[node][1] - pos['start'][1]  for node in pos))

    # Calculate the line coordinates
    x_start = pos['start'][0]
    y_start = pos['start'][1]
    x_end_1 = x_start + length * np.cos(np.deg2rad(angle_1))
    y_end_1 = y_start + length * np.sin(np.deg2rad(angle_1))
    x_end_2 = x_start + length * np.cos(np.deg2rad(angle_2))
    y_end_2 = y_start + length * np.sin(np.deg2rad(angle_2))
    x_end_3 = x_start + length * np.cos(np.deg2rad(angle_3))
    y_end_3 = y_start + length * np.sin(np.deg2rad(angle_3))

    # Line 1: 60 degrees
    plt.plot([x_start, x_end_1], [y_start, y_end_1], 'k-', linewidth=2)

    # Line 2: 180 degrees
    plt.plot([x_start, x_end_2], [y_start, y_end_2], 'k-', linewidth=2)

    # Line 3: 300 degrees
    plt.plot([x_start, x_end_3], [y_start, y_end_3], 'k-', linewidth=2)

    plt.show()
    
def plot_zipf_dist(counts, p_type='bar'):
    zipf_df=pd.DataFrame.from_dict(dict(counts.most_common(50)),orient='index').reset_index()
    zipf_df=zipf_df.rename(columns={'index':'Word', 0:'Count'})

    fig , ax = plt.subplots(figsize=(15,9))
    plt.xticks(rotation=70)
    if p_type=='bar':
        zip_p = sns.barplot(x=zipf_df["Word"],y=zipf_df["Count"])
    if p_type=='line':
        zip_p = sns.lineplot(x=zipf_df["Word"],y=zipf_df["Count"])
    # get label text
    _, xlabels = plt.xticks()
    _, ylabels = plt.yticks()

    # set the x-labels with
    ax.set_xticklabels(xlabels, size=15)
    ax.set_yticklabels(ylabels, size=15)
    ax.set_xlabel("Word",fontsize=20)
    ax.set_ylabel("Count",fontsize=20)
    plt.show()
    
def compare_distributions_in_plot(outcomes,
                                  first_dist, 
                                  second_dist,
                                  first_label,
                                  second_label,
                                  xlabel="Outcomes",
                                  ylabel="Probability",
                                  title="Two distributions"):
    
    x_ticks = list(range(0, len(outcomes)))
    fig, ax = plt.subplots(figsize = (10,7))
    plt.plot(first_dist, label = first_label)
    plt.plot(second_dist, '-.', label = second_label)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(x_ticks, labels = outcomes)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    ax.locator_params(axis='both', nbins=20)
    plt.title(title, fontsize=16)
    plt.show()

def plot_bernoulli(counts):
    fig, ax = plt.subplots(figsize=(8, 6))

    outcomes = ['heads', 'tails']
    bar_labels = ['heads', 'tails']
    bar_colors = ['tab:red', 'tab:blue']

    ax.bar(outcomes, counts, label=bar_labels, color=bar_colors)
    ax.set_xlabel('Outcome', fontsize=16)
    ax.set_ylabel('Number of tosses', fontsize=16)
    ax.set_title(f'Results from {sum(counts)} coin tosses.', fontsize=20)
    ax.legend(title='Coin toss', fontsize=12, title_fontsize=14)
   
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    
def plot_discrete_distribution(df, x_col, title, suptitle, x_lab, y_lab):
    p = sns.displot(data=df, x=x_col, kind="hist", height=6, aspect=1, discrete=True)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title(title, fontsize=16)
    plt.suptitle(suptitle)
    plt.xlabel(x_lab,fontsize=14)
    plt.ylabel(y_lab,fontsize=14)
    plt.show()
    
def plot_entropy(coin_entropies, max_entr=False):
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 1, 100), coin_entropies, color='blue')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title("Entropy of a coin toss", fontsize=16, y=1.05, x=0.145)
    plt.suptitle("I.e. how many bits do we need  on average to encode the distribution",
                verticalalignment='bottom', horizontalalignment='right', x=0.64, y=.9, )
    plt.xlabel("Probability of heads",fontsize=14)
    plt.ylabel("Entropy (average bits)",fontsize=14)
    if max_entr:
        plt.plot([0.5], [1], 
                 marker="o", 
                 markersize=15, 
                 markeredgecolor="red",
                 markerfacecolor="red")
        plt.annotate('max entropy',
                 (0.5,1), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-20), # distance from text to points (x,y)
                 ha='center')
    ax.locator_params(axis='both', nbins=20)
    plt.grid()
    plt.show()
    
def set_grad_plot_fig_ax():
    fig, ax = plt.subplots(figsize=(15,9))
    return fig, ax

def set_grad_plot_styles(ax, xlim, ylim, title):
    ax.locator_params(axis='both', nbins=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('f(x)',fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid()
    
# objective function
def f_polynomial(x):
    return x**2.0
 
# derivative of objective function
def f_polynomial_gradient(x):
    return x * 2.0    

def plot_function(xs, ys, xlim, ylim, title, linewidth=2):
    fig, ax = set_grad_plot_fig_ax()
    set_grad_plot_styles(ax, xlim, ylim, title)
    plt.plot(xs, ys, color='black', linewidth=linewidth)
    
def grad_line(x, x1, y1,f_polynomial_gradient):
    # Define tangent line
    # y = m*(x - x1) + y1
    return f_polynomial_gradient(x1)*(x - x1) + y1

def plot_tangent_at_points(xs, ys,f_polynomial,f_polynomial_gradient,x_vals):
    plot_function(xs, ys, [-5,5], [-1,25], 'Plot of tangent lines at -4, 0, and 4',1)
    for x1 in x_vals:
        y1 = f_polynomial(x1)
        # Define x data range for tangent line
        xrange = np.linspace(x1-.5, x1+.5, 10)
        plt.scatter(x1, y1, color='red', s=50)
        plt.plot(xrange, grad_line(xrange, x1, y1,f_polynomial_gradient), '--', color='red',linewidth = 3)

    plt.show()
    
def tangent_plot(x1):
    # Choose point to plot tangent line
    xs = np.linspace(-10, 10, 1000)
    ys = f_polynomial(xs)
    y1 = f_polynomial(x1)
    xrange = np.linspace(x1-2, x1+2, 10)
    plot_function(xs, ys, [-15, 15], [-20, 115], "Tangents for x from -8 to 8")
    plt.scatter(x1, y1, color='red', s=50)
    plt.plot(xrange, grad_line(xrange, x1, y1, f_polynomial_gradient), '--', color='red', linewidth = 2)
    
def set_grad_plot_styles(ax, xlim, ylim, title):
    ax.locator_params(axis='both', nbins=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('f(x)',fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(alpha=0.5)
    
def set_grad_start(xs, ys):
    plt.plot([xs[0]], [ys[0]], 
                     marker="o", 
                     markersize=15, 
                     markeredgecolor="red",
                     markerfacecolor="red")
    plt.annotate('start',
             (xs[0], ys[0]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
             xytext=(20,-20), # distance from text to points (x,y)
             ha='center')
    
def set_grad_stop(xs, ys):
    plt.plot([xs[-1]], [ys[-1]], 
                 marker="o", 
                 markersize=15, 
                 markeredgecolor="red",
                 markerfacecolor="red")
    plt.annotate('stop',
             (xs[-1], ys[-1]), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
             xytext=(20,-20), # distance from text to points (x,y)
             ha='center')
    
def plot_gradient_descent(xs, f, parabola_x, xlim, ylim, title):
    fig, ax = set_grad_plot_fig_ax()
    ys = []
    for x in xs:
        ys.append(f(x))
    
    parabola_y = f(parabola_x)
    plt.plot(parabola_x, parabola_y, color='black', linewidth=3, alpha=0.5)
    plt.scatter(xs, ys, color='red')
    plt.plot(xs, ys, color='red', linestyle='--', linewidth=2)
    set_grad_plot_styles(ax, xlim, ylim, title)
    set_grad_start(xs, ys)
    set_grad_stop(xs, ys)
    plt.show()