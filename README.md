# GAMES MASTER

This is a [Lewagon](https://www.lewagon.com) final project, with [Camila-mallmann](https://github.com/Camila-mallmann), [abdl242](https://github.com/abdl242) and [myself](https://github.com/Hip-po)

## The project

The goal is to create an AI that learns from nothing how to play different gyms using Reinforcement Learning.

We decided to start with an easy rac car game to understand and master RL theory and pytorch.

Then we will apply our new knowledges on other games.

## Racing car

This game is pretty simple. All the AI has to do is to follow a race track.

As long as the car follow the roads, it betters its reward. But if it does nothing or goes off track, the score is penalized.

To reach the best possible score, we tried different model and way to make our AI learn.

### The model

The agent is learning through a batch of 128 observations of the game we can define as images, array. The model transform the batch to a tensor, and may transform the images into B&W image and crop.

The agent is motivated by a reward it gets as he is moving forward and follow the track.

However, if he stays at the same place or goes off track, he will get penalized and the game should restart if the score is very low.

From the beginning, it will choose randomly an action but the more it is learning the less it will do it.

### <img src="GIF\nathan_driving.gif" width="450px">



### discreet values:

Discrete values choices are the simplest way to learn for a machine as we only choosing between 5 movements and not a range of a multi-dimensional vector.

The model choose one action from the 5 available and get rewarded or penalized.


#### Deep-Q Network

### <img src="GIF\car_racing_dqn_discret_v1.gif" width="450px">

## Mario Bros

Everyone knows Mario Bros except our agent.

First goal here will be training an agent to learn how to play to the game and finish it.

The goal of the game is surviving by stamping toads and turtles and go straight to the end the level and get the flag.

From the environment, we get an observation as an array where 0 is the background, 1 are blocks, 2 are ennemies and 3 the player Mario.


### How the agent works


The agent is treating batch of images as described and evaluate each observation to choose the best decision according to it trhough the history of rewards.

#### Reward system

For this game, we had to initiate ourself the reward to push the agent forward and to make somehow a timecounter importance.

We also penalize the agent if it's not moving his *** or if it dies.

But if it captured the flag, he will be rewarded as a veteran.

#### Observation treatment

The agent is learns through a simplified image we can define as an array.

Through many Conv2D layers, a flatten one and a linear corresponding to the choices.

The only tranformation done here is a to_tensor one to make it readable.

Also, we made a slighty modification to the choices to make our model more efficient, we have added two combined keyboards choice.


## Training

For the moment, we are only training the model on the first level.



## Result
