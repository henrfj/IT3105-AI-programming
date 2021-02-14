# Todo list

## SW

- Reward function needs tweaking. Negative rewards?

## Read further

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

3. Association table / dictionary between state (key) and value (value).
4. NN mapping states to values, implemented in tensorflow deep-learning package.

### Function approximation notes

From the AC-document. TODO:

- Replace critic table V with a NN. Make it general so same leraning algorithm can be used.
- Input of NN is a 1D table. Make a converter.
- 

### FROM Q&A 08.02.2021

1. Q(s,a) vs V(s).

- using Q you are directly making the policy; making value and policy at the same time almost.
- Q fits better in to the off-policy of Q learning.
- But V(s) is just as simple for TD learning. Q has bigger table, but V requires more implementation.

2. Getting started with NN implementation.

- Make a few, and run it for different approxes.
- Need only 1 hidden layer, no advanced techniques. **Copy from a simple net online**.
- Using to learn KERAS is the real problem. Get used to it by approxing some other function.

3. Convergence problem.

- As we are doing random search, finding a single solution is lucky.
- Mess with parameters, to help out the algorithm to converge to.
- Learning rate, discount factor, epsilon, etc.

4. **SplitGD or directly change weight, delta rule with eligibility.**

- If its possible to directly change weights, you are good to go.
- But SplitGD allows for you to be able to open up "apply gradient", allowing you to modify the weight update.
- Use SplitGD if you don't know how to modify "manually".

5. 6x6 diamond convergence.

- Not required for the demo. It should be able to run on 6x6.

6. **Handle inputs and targets in NN**

- input: the state
- target: updated, evaluation of the state.
- learning:
=> We make it into supervised, by using next state to evaluate previous state- to see if our selection made sense.
=> Will get a lot of pairs, (states,actions) - we can then use all the episodes' pairs to fit: "keras.fit()" with split GD.
- So only run NN after you have saved up a bunch of cases: **an episode**
- The target in training NN is no gold standard, just an assumed potential, based on the next state we discover after makin the move.

7. Never ending episode

- For control systems, episode never ends - and we don't reset system like here.

8. Eligibility in NN:

- Should we calculate e_i <- e_i + dV/dw_i,
- or should we use same as for table-method...

9. We do not use validation data: all data is training data.

- The code is just to illustrate breaking up fit routine, ignore the rest.

10. In splitGD, for parameter V_frac, set it to 0.0.

- Might be other things you have to do.
- He uses minibatches, sending entire lists.

11. **Modify_gradient function** => what to do?

- NN approxes function. For each wi in NN, it needs to be modified. apply_gradient in KERAS does this, based on the gradient.
- in splitGD, line 46 calculates the gradients. Then modify these gradients to include e-trace, so we split and modify before KERAS is allowed to apply it onto the weights.
- Normally in NN, gradients is the whole basis needed to modify w's, and you don't need to modify them. Keras.fit() would compute and apply gradients as one atomic operation.
- The formula: (just under equation 20 in a-c.pdf.). wi <- wi + alpha * delta * e_i. Eligibility decays (0.9). We **actually** update e_i based on gradient and eligibility decay, and then use eligibility to update w_i!
- Tape.gradient() is KERAS function to give you the derivatives needed to update eligibilities; the basic gradient.

12. **Sequential model** NN is just fine. Adding layers sequentially, as shown in KERAS tutorial.

13. Can you split **Q-learning into actor and critic**?

- AC concept has different definitions, and the framework can be used for different frameworks.
- So, no reason why Q learning cannot be combined with it. 