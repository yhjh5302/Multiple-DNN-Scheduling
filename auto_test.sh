for servers in 0 2 4 6 8 10
  do
    echo "auto test server: ${servers}"
    python3 main.py --server_low=$servers --server_high=$servers --service_low=6 --service_high=6 --bandwidth_ratio=1.0;
done

for services in 1 2 3 4 5 6 7 8 9
  do
    echo "auto test server: ${services}"
    python3 main.py --server_low=6 --server_high=6 --service_low=$services --service_high=$services --bandwidth_ratio=1.0;
done

for bandwidth in 0.1 0.5 1.0 2.0 3.0
  do
    echo "auto test bandwidth ratio: ${bandwidth}"
    python3 main.py --server_low=6 --server_high=6 --service_low=6 --service_high=6 --bandwidth_ratio=$bandwidth;
done