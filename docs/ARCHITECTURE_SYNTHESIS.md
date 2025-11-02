# Texas 42 Architecture: Core Components Synthesis

**Note**: This document has been superseded by [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md), which provides a more comprehensive view of architectural principles, design philosophy, and mental models.

**For new readers**, please see:
- **Vision**: [VISION.md](VISION.md) - Strategic direction and north star outcomes
- **Orientation**: [ORIENTATION.md](ORIENTATION.md) - Developer onboarding and navigation guide
- **Principles**: [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md) - Design philosophy and mental models
- **Reference**: [CONCEPTS.md](CONCEPTS.md) - Complete implementation reference

---

## Quick Summary (for reference)

### Fundamental Pattern
```
STATE → ACTION → NEW STATE
```

### Two-Level Composition System
```
LAYERS (execution rules) × VARIANTS (action transformation) = Game Configuration
```

### Three-Tier Authority Structure
```
KERNEL → SERVER → TRANSPORT
(logic)  (orchestration)  (routing)
```

### Client-Server Architecture
```
SERVER (authoritative) ← PROTOCOL → CLIENT (delegating)
```

---

**For complete information, see [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md)**
