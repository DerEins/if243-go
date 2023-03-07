import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores,win,reward):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(reward,"g")
    plt.plot(win[1:],"r")
    plt.ylim(bottom=-105,top=105)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(win)-2, win[-1], str(win[-1]))
    plt.text(len(reward)-1, reward[-1], str(reward[-1]))
    plt.show(block=False)
    plt.pause(.1)
