import matplotlib.pyplot as plt


class draw_graph():
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)
        self.line, = self.ax1.plot([], [])
        self.ax1.set_xlabel('batch #')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Loss evolution')
        self.line2, = self.ax2.plot([], [])
        self.ax2.set_xlabel('batch #')
        self.ax2.set_ylabel('Reward')
        self.ax2.set_title('Reward evolution')
        plt.tight_layout()

        self.loss_evolution = []
        self.frame_step = []
        self.reward_evolution = []

        self.iter = 0

    def draw(self, loss, reward):
        self.iter += 1
        self.loss_evolution.append(float(loss.sum().detach().numpy()))
        self.reward_evolution.append(float(reward.sum().detach().numpy()))
        self.frame_step.append(self.iter)
        print(self.reward_evolution)

        # LOSS
        if len(self.loss_evolution) < 20:
            self.ax1.set_ylim(0, 100000)
            self.ax1.set_xlim(0, 100)
        else:
            self.ax1.set_xlim(len(self.loss_evolution) - 20, len(self.loss_evolution))
            self.ax1.set_ylim(min(self.loss_evolution[-20:]), max(self.loss_evolution[-20:]))
        self.line.set_xdata(list(range(len(self.loss_evolution))))
        self.line.set_ydata(self.loss_evolution)
        self.line.figure.canvas.draw_idle()

        # REWARD
        if len(self.reward_evolution) < 20:
            self.ax2.set_ylim(-50, 50)
            self.ax2.set_xlim(0, 100)
        else:
            self.ax2.set_xlim(len(self.reward_evolution) - 20, len(self.reward_evolution))
            self.ax2.set_ylim(min(self.reward_evolution[-20:]), max(self.reward_evolution[-20:]))
        self.line2.set_xdata(list(range(len(self.reward_evolution))))
        self.line2.set_ydata(self.reward_evolution)
        self.line2.figure.canvas.draw_idle()
