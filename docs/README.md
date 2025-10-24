# Cross-Gym Documentation

Welcome to the Cross-Gym documentation!

## 📚 Documentation Structure

### For Users

1. **[README.md](../README.md)** - Start here!
   - Project overview
   - Quick start guide
   - Installation instructions
   - Feature highlights

2. **[GETTING_STARTED.md](../GETTING_STARTED.md)** - Step-by-step tutorial
   - Core concepts
   - Your first task
   - Configuration tips
   - Examples

3. **[examples/README.md](../examples/README.md)** - Example tasks
   - Simple task example
   - How to create your own task

### For Developers

4. **[DESIGN.md](../DESIGN.md)** - Design philosophy
   - Why Cross-Gym exists
   - Framework analysis (direct, manager_based, isaaclab)
   - Key architectural decisions
   - Implementation plan

5. **[ARCHITECTURE.md](../ARCHITECTURE.md)** - Technical architecture
   - Component diagrams
   - Design patterns
   - Data flow diagrams
   - Import patterns
   - Cross-simulator compatibility

6. **[IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)** - Progress tracking
   - Component-by-component status
   - Completion percentages
   - Next steps
   - Known issues

---

## 🎯 Quick Navigation

### I want to...

**...understand what Cross-Gym is**
→ Start with [README.md](../README.md)

**...create my first task**
→ Follow [GETTING_STARTED.md](../GETTING_STARTED.md)

**...see example code**
→ Check [examples/](../examples/)

**...understand the design**
→ Read [DESIGN.md](../DESIGN.md)

**...see technical details**
→ Check [ARCHITECTURE.md](../ARCHITECTURE.md)

**...know what's implemented**
→ See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)

**...contribute to the project**
→ Read [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) for what needs work

---

## 📖 Document Purposes

| Document | Purpose | Audience |
|----------|---------|----------|
| README.md | Project overview & quick start | Everyone |
| GETTING_STARTED.md | Hands-on tutorial | Users |
| DESIGN.md | Design philosophy & rationale | Developers |
| ARCHITECTURE.md | Technical implementation details | Developers |
| IMPLEMENTATION_STATUS.md | Progress tracking | Contributors |
| examples/ | Working code examples | Users |

---

## 🚀 Recommended Reading Order

### For New Users:
1. README.md
2. GETTING_STARTED.md
3. examples/README.md
4. examples/simple_task_example.py

### For Developers:
1. README.md
2. DESIGN.md
3. ARCHITECTURE.md
4. IMPLEMENTATION_STATUS.md
5. Explore source code in `cross_gym/`

### For Contributors:
1. IMPLEMENTATION_STATUS.md (see what needs work)
2. ARCHITECTURE.md (understand the patterns)
3. DESIGN.md (understand the philosophy)
4. Contribute! 🎉

---

## 🔍 Key Concepts Explained

### Simulator Abstraction
Cross-Gym separates simulator-specific code from RL logic. One configuration, multiple simulators.

### Manager Pattern
MDP components (observations, rewards, actions) are managed by specialized classes that compose together.

### Backend View Pattern
Assets hold data, delegate operations to simulator-specific backends. Clean separation.

### Configuration-Driven
Everything is configurable via dataclasses. Zero hard-coded parameters.

---

## 💡 Learning Path

```
START → README.md → GETTING_STARTED.md → Build your first task
           ↓
      DESIGN.md → Understand philosophy
           ↓
    ARCHITECTURE.md → Learn patterns
           ↓
IMPLEMENTATION_STATUS.md → See what's possible
           ↓
      Contribute to Cross-Gym! 🚀
```

---

## 📝 Documentation Guidelines

When contributing documentation:
- Keep README.md as the main entry point
- Update GETTING_STARTED.md with new features
- Document design decisions in DESIGN.md
- Add technical details to ARCHITECTURE.md
- Track progress in IMPLEMENTATION_STATUS.md
- Add examples in examples/

---

## 🤝 Get Help

- **Questions?** Open a GitHub issue
- **Bug reports?** Check IMPLEMENTATION_STATUS.md known issues first
- **Feature requests?** Review DESIGN.md to understand the philosophy

---

**Happy coding! 🤖**

