# Todo list

### SW class:
- Find out what kind of *action* I want to pass: Maybe a tuple or just the integer arrays of child states?
- 



## RL engine:


## Read further:
1. Does the actor chose first, and is then ecaluated? Or will the actor ask the critic before acting.(*)
=> Seems like the actor chooses action, then state1 and state2 are passed to critic, who criticises the transition.

### ACTOR
- Try to implement as simple state selection first V(s), before SAPs
- On-policy, epsilon greedy.
 => initially constant epsiolon, but i could possibly decrease as we get closer to target behaviour.
- Policy; represented as array or python - dictionary, mapping all possible state-action pairs to desierability of the action. 
=> normalize desierability across all legal actions from s, yielding a probability distribution. 
- The policty itself is Contained within actor and Stored as a dictionary - (state1): [(state2_1, value), (state2_2, value), ...]



### CRITIC
Value function: Contained and updated by critic. 
1. Using V(s): Is a python dictionary: Key is state, returns value of state. Actor passes state to critic(*), who returns calculated TD error for given state.
2. Using Q(S,a): Python dictionary: key is touple of (state1, state2) - where state2 is a legal transition state. 
=> V(s) is basically the sum of all Q's?
=>Use eligability traces to updates this function quicker.
Two different implementations of value function is required.
1. Association table / dictionary between state (key) and value (value). 
2. NN mapping states to values, implemented in tensorflow deep-learning package. 

