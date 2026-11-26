# This project's edge cases

The `test-new-paths` skill reads this file. It is this project's part of the skill (#6631).

Each of these that a change can meet needs a test:

- a dead pole and a default-on pole.
- both poles of a neuron, positive and negative.
- every kind of unit that the code accepts: a neuron pole, an attention head, a dictionary feature.
- an old record with no value for a new field. Records are write-once, so old records stay in the registry.
