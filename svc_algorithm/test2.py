import algorithm_sim
import requester
import sim_system

if __name__ == "__main__":
    r_sim = requester.RequestSim(1)
    r_sim.init(10, 2)
    r_sim.sim_start()
    r_sim.env.run(until=600)
    request = r_sim.result

    sim = sim_system.SimulatorPack(end_t=600)
    rl = algorithm_sim.RL(sim)
    rl.train()


