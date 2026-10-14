# TLCS — Deep Q-Learning Agent for Traffic Signal Control

A Deep Q-Learning agent that controls the traffic light of a single 4-way
intersection in the [SUMO](https://eclipse.dev/sumo/) traffic simulator, choosing
one of four green phases at each decision step to reduce vehicle delay.

## Credits

This project is derived from **TLCS** by Andrea Vidali,
["Deep Q-Learning Agent for Traffic Signal Control"](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control)
(MIT License, Copyright (c) 2019 Andrea Vidali). The intersection network, the
state discretisation and the original DQN training loop come from that project.
The changes made here include the shared metric definitions, the fixed-time
baseline comparison, the saturation guard and the libsumo/subscription data
path; see the git history.

## License

MIT. See [LICENSE](LICENSE), which names both copyright holders.
