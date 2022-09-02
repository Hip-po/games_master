import games_master.run_model.car_racing_DQN as DQN_v1
import games_master.run_model.car_racing_DQN_v2 as DQN_v2
import human_play as human
from tkinter import *

fenetre = Tk()

def run_v1():
    human_rwd=human.run()
    ia_rwd=DQN_v1.run()

    Label(fenetre, text=f"ia score: {int(ia_rwd)}         your score: {int(human_rwd)}").pack()

def run_v2():
    human_rwd=human.run()
    ia_rwd=DQN_v2.run()

    Label(fenetre, text=f"ia score: {int(ia_rwd)}         your score: {int(human_rwd)}").pack()

label = Label(fenetre, text="Car racing")
label = Label(fenetre, text="Chose who to face",width=40)
label.pack()
Button(fenetre, text="Deep-Q Neuralnetwork vesrion 1", command=run_v1).pack()
Button(fenetre, text="Deep-Q Neuralnetwork version 2", command=run_v2).pack()
Button(fenetre, text="EXIT", command=fenetre.destroy).pack()


fenetre.mainloop()
