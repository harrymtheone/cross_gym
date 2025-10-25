# Cross-Gym Documentation

Welcome to the Cross-Gym documentation! This directory contains comprehensive guides for users and developers.

---

## 🎯 Quick Navigation

| Question                         | Document                                           |
|----------------------------------|----------------------------------------------------|
| What is Cross-Gym?               | [../README.md](../README.md)                       |
| How do I start?                  | [../GETTING_STARTED.md](GETTING_STARTED.md)        |
| How do examples work?            | [../examples/README.md](../examples/README.md)     |
| What design decisions were made? | [IMPROVEMENTS.md](IMPROVEMENTS.md)                 |
| How do I configure simulators?   | [SIMULATOR_CONFIGS.md](SIMULATOR_CONFIGS.md)       |
| How does architecture work?      | [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) |
| What's implemented?              | [STATUS.md](STATUS.md)                             |
| Is it complete?                  | [CHECKLIST.md](CHECKLIST.md)                       |

---

## 📚 Documentation Index

### For New Users (Start Here!)

1. **[../README.md](../README.md)** - Main project overview
    - What is Cross-Gym?
    - Quick start guide
    - Feature highlights
    - Installation

2. **[../GETTING_STARTED.md](GETTING_STARTED.md)** - Hands-on tutorial
    - Core concepts explained
    - Step-by-step first task
    - Configuration examples
    - Troubleshooting

3. **[../examples/README.md](../examples/README.md)** - Example tasks
    - Simple task example
    - How to create your own task
    - Switching simulators

---

### For Framework Developers

4. **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Design decisions & improvements
    - configclass implementation
    - Simulator-specific configs
    - Quaternion convention
    - Python 3.8+ compatibility
    - All design improvements explained

5. **[SIMULATOR_CONFIGS.md](SIMULATOR_CONFIGS.md)** - Simulator configuration guide
    - Why simulator-specific configs?
    - IsaacGymCfg usage
    - GenesisCfg usage
    - How to switch simulators
    - Benefits of the pattern

6. **[NEW_SIM_PATTERN.md](NEW_SIM_PATTERN.md)** - Detailed pattern explanation
    - Problems with super-sets
    - The elegant solution
    - How it works internally
    - Comparison table

7. **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - Visual architecture
    - Overall architecture diagram
    - Data flow diagrams
    - Simulator integration
    - Manager composition
    - Module dependencies

---

### Reference Documentation

8. **[FRAMEWORK_COMPLETE.md](FRAMEWORK_COMPLETE.md)** - Implementation summary
    - What was built
    - Statistics
    - How to use
    - Key achievements

9. **[STATUS.md](STATUS.md)** - Current status
    - Implementation status
    - Statistics
    - Capabilities
    - Next steps

10. **[CHECKLIST.md](CHECKLIST.md)** - Verification checklist
    - All implemented features
    - Design excellence checkpoints
    - Verification questions

---

### Conventions & Guides

11. **[QUATERNION_CONVENTION.md](QUATERNION_CONVENTION.md)** (if exists) - Quaternion format
    - Why (w, x, y, z)?
    - Identity quaternion
    - Automatic conversion
    - Examples

---

## 🎯 Quick Navigation

### I want to...

**...understand what Cross-Gym is**  
→ Start with [../README.md](../README.md)

**...create my first task**  
→ Follow [../GETTING_STARTED.md](GETTING_STARTED.md)

**...see working code**  
→ Check [../examples/](../examples/)

**...understand design decisions**  
→ Read [IMPROVEMENTS.md](IMPROVEMENTS.md)

**...configure simulators**  
→ See [SIMULATOR_CONFIGS.md](SIMULATOR_CONFIGS.md)

**...see the architecture**  
→ Check [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)

**...know what's implemented**  
→ See [STATUS.md](STATUS.md) or [FRAMEWORK_COMPLETE.md](FRAMEWORK_COMPLETE.md)

**...verify completeness**  
→ Review [CHECKLIST.md](CHECKLIST.md)

---

## 📖 Recommended Reading Order

### For New Users:

```
README.md → GETTING_STARTED.md → examples/README.md → Build your task!
```

### For Developers:

```
README.md → IMPROVEMENTS.md → ARCHITECTURE_DIAGRAM.md → Explore code
```

### For Contributors:

```
STATUS.md → CHECKLIST.md → Pick a task → Contribute!
```

---

## 📁 Documentation Structure

```
cross_gym/
├── README.md                        # Main entry point (ROOT)
├── GETTING_STARTED.md               # Tutorial (ROOT for easy access)
│
├── docs/                            # All other documentation
│   ├── README.md                    # This file - documentation index
│   ├── IMPROVEMENTS.md              # Design decisions
│   ├── SIMULATOR_CONFIGS.md         # Simulator config guide
│   ├── NEW_SIM_PATTERN.md           # Pattern explanation
│   ├── ARCHITECTURE_DIAGRAM.md      # Visual diagrams
│   ├── FRAMEWORK_COMPLETE.md        # Implementation summary
│   ├── STATUS.md                    # Current status
│   ├── CHECKLIST.md                 # Verification checklist
│   └── QUATERNION_CONVENTION.md     # Quaternion guide (if exists)
│
└── examples/
    └── README.md                    # Example guide
```

---

## 🎨 Document Purposes

| Document                    | Purpose          | Audience     | Location  |
|-----------------------------|------------------|--------------|-----------|
| **README.md**               | Project overview | Everyone     | Root      |
| **GETTING_STARTED.md**      | Tutorial         | Users        | Root      |
| **examples/README.md**      | Examples         | Users        | examples/ |
| **IMPROVEMENTS.md**         | Design decisions | Developers   | docs/     |
| **SIMULATOR_CONFIGS.md**    | Sim config guide | Developers   | docs/     |
| **NEW_SIM_PATTERN.md**      | Pattern details  | Developers   | docs/     |
| **ARCHITECTURE_DIAGRAM.md** | Visual arch      | Developers   | docs/     |
| **FRAMEWORK_COMPLETE.md**   | Summary          | Contributors | docs/     |
| **STATUS.md**               | Current status   | Contributors | docs/     |
| **CHECKLIST.md**            | Verification     | Contributors | docs/     |

---

## 💡 No Duplicates!

Each document has a **single, clear purpose**:

- No overlapping content
- No redundant information
- Clear separation of concerns
- Easy to find what you need

---

## 🔍 Finding Information

**Question** → **Document**

- What is Cross-Gym? → README.md
- How do I start? → GETTING_STARTED.md
- How do I configure simulators? → SIMULATOR_CONFIGS.md
- Why (w,x,y,z) quaternions? → QUATERNION_CONVENTION.md
- What's been implemented? → STATUS.md or FRAMEWORK_COMPLETE.md
- How does it work internally? → ARCHITECTURE_DIAGRAM.md
- What design decisions were made? → IMPROVEMENTS.md
- Is it complete? → CHECKLIST.md

---

**All documentation is organized, current, and non-redundant!** ✨

