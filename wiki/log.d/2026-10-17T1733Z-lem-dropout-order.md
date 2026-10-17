## [2026-10-17 | lem | the drop-out order (#196): cleaner to 12 neurons over three seeds; the last words are the trick position]

[[lem-dropout-order-2026-10-17]]: 21 trained and 21 untrained nets at seven widths. Narrower is
cleaner down to 12 neurons (0.633 at 192 → 0.759 at 24 → 0.735 at 12, floors 0.17–0.23). The
drop-out order: `lead` and `late` last, then the trick in front of the mover and the mover's
trump, then count and trump on the table; mastery outside trump needs 136 neurons. Predictions
held 3 of 5: two neurons close 0.28, not 0.40, and two of three 4×2 nets failed to learn and fell
to the floor. Scope: this net, these inputs, list v1. The first two friction logs on the
clean-neuron skill came from this grade.
