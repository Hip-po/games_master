from config import CFG
import gym


def manual(env, agt):
    for _ in range(100):
        old_obs = new_obs
        action = CFG.dict_choice['z']

        old_obs = new_obs

        new_obs, reward, done, _ = env.step(action)
        agt.agent_step(old_obs, action, new_obs, reward)

        if done:
            env.reset()

    for _ in range(500):
        input_choice = input()
        old_obs = new_obs

        if len(input_choice) <= 1:

            if input_choice not in CFG.dict_choice.keys():
                action = 0

                old_obs = new_obs

                new_obs, reward, done, info = env.step(action)
                agt.agent_step(old_obs, action, new_obs, reward)
                if done:
                    env.reset()

            else:

                action = CFG.dict_choice[input_choice]
                old_obs = new_obs

                new_obs, reward, done, info = env.step(action)
                agt.agent_step(old_obs, action, new_obs, reward)
                if done:
                    env.reset()

        else:
            try:
                n = int(input_choice[:-1])
                letter = input_choice[-1]
            except:
                n = 5
                letter = ' '
            for _ in range(n):
                try:
                    action = CFG.dict_choice[letter]

                    old_obs = new_obs

                    new_obs, reward, done, info = env.step(action)
                    agt.agent_step(old_obs, action, new_obs, reward)
                    if done:
                        env.reset()
                except:
                    action = CFG.dict_choice['z']

                    old_obs = new_obs

                    new_obs, reward, done, info = env.step(action)
                    agt.agent_step(old_obs, action, new_obs, reward)
                    if done:
                        env.reset()

        if not CFG.VM:
            env.render()

        if done:
            env.reset()
