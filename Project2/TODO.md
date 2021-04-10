# Ask studass: 
- TOPP:
    --> 1. problem er at to agenter som spiller mot hverandre er helt deterministiske. Hvis spiller en vinner når han starter, så gjør han alltid det.
    --> 2. Enten: Legge til litt random moves, se hvem som er best til å håndtere alle situasjoner, eller bruke fordelingen direkte til å velge moves.


# This has worked!
- Cooking right now:
==> output: 
==> T1: 
==> T2: 
==> T3: Iron man 2 - with alternating starting players and improved training regime.

- Ideas: lower learning rate. Only patterns that keeps repeating will be recognized!
        => Using mbs param! Dont do say 100 epochs of one mbs! Do 100 different mbs from RBUF!

- TOPP idea: as we have some stochasticity in the TOPP, the early ones need to have seen /train on way less data - less general.
        => use small mbs, but biig RBUF - the later ones are frequently training on a lot of different scenarioes. 
        => Thicker NN so that its easier to overfit with small data sizes.

# TODO list
Things to do.

1. Test the power model vs the best models made in by RL_2. Make a "power model" NN style.

2. Test more on the alternating players thing. Is it really good or not?

3. Try to **remove buffer** and only fit on the newest data. Overfit AF if needed.
=> Did this, got waay to specific.

4. Try to wait until we have enough data, and train reasonably well on it as we go.
    => Works well, **but** if we have enoug data, we might want to train "harder" on it (more epochs, especially last training!)

# OHT MODELS:

1. The deterministic power models

2. The deterministic iron man model

3. The epsilon greedy model

4. The network that changes strategy (model) based on performance!

5. Tweak an retry with the best one.

# Q&A

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
