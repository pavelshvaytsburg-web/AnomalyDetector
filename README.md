# AnomalyDetector
```
python main.py --input ./data/network_traffic.csv --output outputs \
  --combine_rule or \
  --p 10 --thr_combine lenient \
  --window 5 --freq_key dst_port --rarity_q 0.05 --burst_q 0.99 \
  --plots
  ```
  