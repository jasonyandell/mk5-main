# Stage 0 exposure Scan B: decision-level harm

- device: `mps`
- nonzero-mass candidate states (from Scan A): 200
- states evaluated: 200
- argmax flips (production encoding): 20

## Worst 10 by exact regret of legacy choice (Q)

| seed | depth | regret Q | max shift Q | argmax flip |
|---|---:|---:|---:|---:|
| 902512 | 20 | 7.370 | 7.242 | yes |
| 901768 | 20 | 5.267 | 27.088 | yes |
| 903929 | 20 | 5.241 | 6.904 | yes |
| 905134 | 20 | 4.274 | 16.958 | yes |
| 905233 | 16 | 2.229 | 4.527 | yes |
| 906558 | 16 | 1.532 | 3.339 | yes |
| 905265 | 16 | 1.047 | 15.378 | yes |
| 904506 | 16 | 0.723 | 5.434 | yes |
| 904280 | 16 | 0.682 | 1.971 | yes |
| 903094 | 16 | 0.680 | 1.595 | yes |

## Worst 10 by max abs action shift (Q)

| seed | depth | max shift Q | regret Q | argmax flip |
|---|---:|---:|---:|---:|
| 901768 | 20 | 27.088 | 5.267 | yes |
| 906202 | 20 | 22.687 | 0.000 | no |
| 905887 | 20 | 21.052 | 0.000 | no |
| 904539 | 20 | 17.256 | 0.000 | no |
| 905134 | 20 | 16.958 | 4.274 | yes |
| 904156 | 20 | 16.529 | 0.000 | no |
| 900123 | 20 | 16.481 | 0.000 | no |
| 900954 | 20 | 15.976 | 0.000 | no |
| 901648 | 20 | 15.441 | 0.000 | no |
| 905265 | 16 | 15.378 | 1.047 | yes |
