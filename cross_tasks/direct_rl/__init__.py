"""Direct RL tasks.

Tasks defined using the direct RL workflow (without managers).
For simpler tasks or when you need more direct control.
"""

from .t1 import t1_tasks

direct_tasks = {}
direct_tasks.update(t1_tasks)
