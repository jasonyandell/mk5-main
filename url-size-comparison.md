# URL Size Comparison

## Example Game with 9 Actions

### Old Format (5117 characters)
```
?data=%7B%22initial%22%3A%7B%22phase%22%3A%22bidding%22%2C%22players%22%3A%5B%7B%22id%22%3A0%2C%22name%22%3A%22Player%201%22%2C%22hand%22%3A%5B%7B%22high%22%3A6%2C%22low%22%3A0%2C%22id%22%3A%226-0%22%7D...
```
(Full GameState JSON + all action details)

### New Format (179 characters)
```
?d=eyJ2IjoxLCJzIjp7InMiOjE3NTM3NTYzOTczNDF9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0NSJ9LHsiaSI6IjY0In0seyJpIjoiNTUifSx7ImkiOiI1MCJ9LHsiaSI6IjQxZCJ9XX0
```

## Compression Techniques Used

1. **Minimal Initial State**: Only store the seed (and non-default values)
   - Before: Full GameState with all players, hands, etc.
   - After: Just `{s: 1753756397341}`

2. **Action Compression**: Map verbose action IDs to short codes
   - `"pass"` → `"p"`
   - `"bid-30"` → `"30"`
   - `"select-trump-5"` → `"t5"`
   - `"play-6-4"` → `"64"`

3. **Base64 Encoding**: More efficient than URL encoding
   - Removes need for percent-encoding special characters
   - URL-safe variant (no +, /, or = padding)

## Size Reduction: 96.5%!

This means:
- Easier to share game links
- Less likely to hit URL length limits
- Faster to load and process
- Still human-readable when decoded