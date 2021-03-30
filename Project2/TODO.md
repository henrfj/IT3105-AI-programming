# TODO list
Todo list and list of interesting questions from Q&As.

1. Child_states is redundant?
    - Cant we just use possible_moves list directly to chose a move?

2. If slow, make it so that we dont save enteties of the board, but simply play backwards and forwards on one board, and only save tokens.
    - 

3. Store and update the hash - value of state so we dont need to compute it every time

# Q&A 15.03.2021

1. MCTS own implementation
- As long as MCTS is apart from NN and game logic its ok. 
- Own classes for nodes and edges, or use dicts for each value (N(s), N(s,a), Q(s,a))

2. State manager = game logic
- Final state, legal moves, general knowledge.
- Can make a **deep copy** of the game instance each time we do a MCTS.
    This is to avoid altering the actual game when doing MCTS searches.

3. After each actual moves, should we keep the MCTS tree constructed by the simulations, or prune it so the root is the current state? 
- The info is not that good in the beginning, he would prune it, and remake the MCT each episode - as the default policy is better now.

4. Actor stochastic?
- In training yes.

5. "I found that using Q instead of visit count gave good results for low number of searches (low M)"
- Important to be exploratory both in tree policy and default policy. 
- Still an actor. Q values are simple ~ basically moves. The option to add critic is seperate, instead of rollout.


6. NO. minibatches
- More is not always better.

7. Sim games vs actual games for training:
- Its smart to use a timer for simulating for any one move (3-4 seconds)
    => The first 4-5-6 moves, will only do like 10 rollouts, but the information in the top is pretty bad anyway.
- Reletively difficult problem: reduced chance of getting lucky by running random episodes - invest more time in the tree, less in actual games!

8. Replay buffer size:
- 512 is used
- Too large: will keep buffers from back in episode 5 while you are in episode 1000. 
    => But often better player will lose to worse player who play more randomly.
    => Agents get better at playing similar oponent, not accounting the dumb stuff. 
- Test your agent against random agents to get actual performance measure: then keep the best.
- "Hall of fame" keeping the best agents of there "time" (no. episodes), then compare them to randoms.

9. **Generating children**:
- First time a leaf is seen, generate all children! => Rollout from one of them!
    => Problem: Biases parents based on children. A good children will make the parent more desirable than it probably is.


10. Rollouts early, how random should they be:
- Use epsilon greedy: 10% --> 0%.
- Huge search spaces. Random all the time, not very focused : won't get enough good info.

11. Augmenting training data:
- Swap p1 and p2 pieces, rotate board etc to create much training data from a few.