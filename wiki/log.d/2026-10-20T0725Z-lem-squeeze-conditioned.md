## [2026-10-20 | lem | the squeeze holds with position out (#227): 24 neurons 0.549 against 192 neurons 0.406]

[[lem-squeeze-conditioned-2026-10-20]]: the 42 drop-out nets re-read conditioned on the play count.
- The field net at 24 neurons is still cleaner than at 192: 0.549 against 0.406 over three seeds,
  with floors of 0.014 and 0.077. Every width from 12 neurons up is 0.33–0.53 above its floor.
- Below 12 neurons the units do the play count's own job, the mover's place in the trick. A
  conditioned read cannot credit that by construction; unconditioned they are far above their
  floor.
- The last words beyond position are the mover's trump: `holds-boss-trump` and `no-trump-held`.
- 4 of 4 predictions held.

CLAUDE.md's sentence goes to the Mayor with the conditioned numbers.
