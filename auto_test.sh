for servers in 6
  do
  for services in 3
    do
      echo "auto test - server: ${servers} service: ${services}"
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="Local" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="Edge" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="HEFT" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="CPOP" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="PEFT" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="PSOGA" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="Genetic" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="MemeticPSOGA" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Layerwise" --offloading="MemeticGenetic" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="Local" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="Edge" --iteration=1;
      python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="HEFT" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="CPOP" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="PEFT" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="Greedy" --iteration=1;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="PSOGA" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="Genetic" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="MemeticPSOGA" --iteration=10;
      # python3 main.py --num_servers=$servers --num_services=$services --bandwidth_ratio=1.0 --partitioning="Piecewise" --offloading="MemeticGenetic" --iteration=10;
  done
done