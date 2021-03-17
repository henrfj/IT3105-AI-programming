# TODO list
Todo list and list of interesting questions from Q&As.
1. Make the Graph rotate:
    - Possible solution: adjacency matrix made from the mesh matrix.

# Q&A 15.03.2021

1. MCTS own implementation
- As long as MCTS is apart from NN and game logic its ok. 
- Own classes for nodes and edges, or use dicts for each value (N(s), N(s,a), Q(s,a))

2. State manager = game logic
- Final state, legal moves, general knowledge.
- Can make a **deep copy** of the game instance each time we do a MCTS.
    This is to avoid altering the actual game when doing MCTS searches.
- 