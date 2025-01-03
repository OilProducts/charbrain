Input block is vanilla transformer.  as new observations arrive they are pushed onto the stack of state.  When the autoencoder that predicts the next state is "surprised" (as measured by loss function) we will encode a "lesson" that is pushed through another autoencoder decoder here just reconstructs state, can therefore be trained independently on every sample, just don't use it as a "lesson" unless surprise happens, also might be appropriate to freeze the encoder prior to training the rest of the system.

the lessons are used/stored and spat out by similarity to the current state (reminded) perhaps similar to rag/vectordb?

the decision module takes what it is being "reminded" of and the current output of the input block (after going through the lesson encoder) and uses that to generate an action. 

Using this setup it will be possible to predict the next state, and generate a new action, repeatedly, and call this a plan.  This plan can be critiqued by the critic module, which will be trained to predict the reward signal, and therefore we can now generate plans that are likely to maximize the reward signal over the planning horizon, which can be arbitrarily long.


Somewhere in here there has to be some kind of reward signal that is propagated back th