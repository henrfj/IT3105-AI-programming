# Todo list

### SW class:
- Find out what kind of *action* I want to pass: Maybe a tuple or just the integer arrays of child states?
- 


### ACTOR
- Try to implement as simple state selection first V(s), before SAPs
- On-policy, epsilon greedy.
 => initially constant epsiolon, but i could possibly decrease as we get closer to target behaviour.
- Policy; represented as array or python - dictionary, mapping all possible state-action pairs to desierability of the action. 
=> normalize desierability across all legal actions from s, yielding a probability distribution. 

### CRITIC
Two different implementations required.
1. Association table / dictionary between state (key) and value (value). 
2. NN mapping states to values, implemented in tensorflow deep-learning package. 
