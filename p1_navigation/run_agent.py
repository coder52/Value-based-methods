from utils import load_agent, load_environment, evaluate_agent

def main():
    ENV_PATH = "Banana_Windows_x86_64/Banana.exe"
    CHECKPOINT_PATH = "checkpoint_dqn_dropout.pth"

    # Load environment and agent
    env, brain_name, _ = load_environment(ENV_PATH, no_graphics=False)
    agent = load_agent(CHECKPOINT_PATH, dueling=False)

    # Evaluate the agent
    score = evaluate_agent(env, brain_name, agent)
    print(f"Score: {score}")

    env.close()


if __name__ == "__main__":
    main()
