# TODO list
Todo list and list of interesting questions from Q&As.

1. **Kan vurdere å bruke -1, 0, 1** i board, og playerID -1 og 1. Akkurat nå jobber vi bare med positive tall.
- Tanh tar høyde for negative verdier. Vi trenger kanskje bare ReLU akkurat nå som vi ikke ønsker negative verdier i input eller output.

2. **If slow**, make it so that we dont save enteties of the board, but simply play backwards and forwards on one board, and only save tokens.
    => DONE
3. **Prune** away useless information in the prune - function (use the children register or something)

4. **Instead** of using the paper definition to update Q values, use the slide version.

5. **Paper version**: Read the paper and see if we can optimize our algorithm.

6. *Test ***MCTS*** by itself*. Some worrying results were seen, where a single path was "locked" onto.

7. Right now **D** has zero values for all impossible moves, which is what we trains towards. When we in mcts use NN to generate moves, we still have to dot the output with legal moves.

8. Mulige problem: Trener flere ganger på den gamle dataen, overfitting.
-  Bruk validation data.
-  Mindre buffer size.

9. D og flat er formattert (1,k^2), men brukes kanskje som om de var (k^2, ) i mcts og rl agent.

## Q&As of interest.

1. Actor stochastic?
- In training yes.

2. "I found that using Q instead of visit count gave good results for low number of searches (low M)"
- Important to be exploratory both in tree policy and default policy. 
- Still an actor. Q values are simple ~ basically moves. The option to add critic is seperate, instead of rollout.

3. NO. minibatches
- More is not always better.

4. Sim games vs actual games for training:
- Its smart to use a timer for simulating for any one move (3-4 seconds)
    => The first 4-5-6 moves, will only do like 10 rollouts, but the information in the top is pretty bad anyway.
- Reletively difficult problem: reduced chance of getting lucky by running random episodes - invest more time in the tree, less in actual games!

5. Replay buffer size:
- 512 is used
- Too large: will keep buffers from back in episode 5 while you are in episode 1000. 
    => But often better player will lose to worse player who play more randomly.
    => Agents get better at playing similar oponent, not accounting the dumb stuff. 
- Test your agent against random agents to get actual performance measure: then keep the best.
- **"Hall of fame"** keeping the best agents of theire "time" (no. episodes), then compare them to randoms.

6. **Generating children**:
- First time a leaf is seen, generate **all** children! => Rollout from one of them!
    => Problem: Can end up biasing parents, where only a good child is explored and the bad once remain unexplored.

7. Rollouts early, how random should they be:
- Use epsilon greedy: 10% --> 0%.
- Huge search spaces. Random all the time, not very focused : won't get enough good info.
