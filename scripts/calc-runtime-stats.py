#!/usr/bin/python3
import sys
import statistics
import json

runtime_arg = sys.argv[1:]
runtime_arr = runtime_arg[0].strip(',').split(',')
runtimes = [float(n) for n in runtime_arr]

output = {}
output['avg'] = statistics.mean(runtimes)
output['med'] = statistics.median(runtimes)
output['max'] = max(runtimes)
output['min'] = min(runtimes)
output['std'] = statistics.stdev(runtimes)
output['_runtimes'] = runtimes

output_json = json.dumps(output, sort_keys=True, indent=4)
print(output_json)
