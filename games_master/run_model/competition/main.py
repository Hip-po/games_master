import games_master.run_model.car_racing_DQN as DQN
import human_play as human
from tkinter import *

fenetre = Tk()

def run():
    human_rwd=human.run()
    ia_rwd=DQN.run()

    Label(fenetre, text=f"ia score: {int(ia_rwd)}         your score: {int(human_rwd)}").pack()
    Button(fenetre, text="try again", command=run).pack()

label = Label(fenetre, text="Car racing")
label = Label(fenetre, text="Chose who to face",width=40)
label.pack()
Button(fenetre, text="Deep-Q Neuralnetwork", command=run).pack()
Button(fenetre, text="EXIT", command=fenetre.destroy).pack()


fenetre.mainloop()
