import random
from lander_env import LanderEnv, STATE_SIZE

env = LanderEnv()
s = env.reset()
assert len(s) == STATE_SIZE, (len(s), STATE_SIZE)
for _ in range(500):
    s, r, d, info = env.step(random.randint(0, 5))
    assert len(s) == STATE_SIZE
    if d:
        s = env.reset()
print("ok, terrain features:", [round(v, 2) for v in s[10:]])